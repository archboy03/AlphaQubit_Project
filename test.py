import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import pickle
import os
from model.model import CycleArchitecture
from data_generation_types.load_hardware_data import (
    load_hardware_data,
    load_baseline_predictions,
)
try:
    from config_local import LocalConfig as Config
except ImportError:
    from config import Config

config = Config()

# --- CONFIGURATION ---
DISTANCE = config.code_distance
HIDDEN_SIZE = config.hidden_size
NUM_LAYERS = config.num_layers
NUM_STABILIZERS = DISTANCE ** 2 - 1

TEST_ROUND = config.test_round
CHECKPOINT_DIR = config.checkpoint_dir
EVAL_BATCH_SIZE = 256

# MoE auxiliary loss coefficient (must match training)
AUX_LOSS_COEF = 0.01


# --- 1. MODEL (identical to train.py / finetune.py) ---
def unroll_model(syndromes):
    """
    syndromes: (Batch, Rounds, NUM_STABILIZERS, 4)
    Returns: (Batch, Rounds, 1), aux_loss
    """
    cycle_model = CycleArchitecture(
        mixing_mult=0.5,
        output_size=HIDDEN_SIZE,
        distance=DISTANCE,
        num_layers=NUM_LAYERS,
    )

    batch_size = syndromes.shape[0]
    d_state = jnp.zeros((batch_size, NUM_STABILIZERS, HIDDEN_SIZE))

    logits_over_time = []
    aux_loss_total = 0.0

    for t in range(syndromes.shape[1]):
        current_check = syndromes[:, t, :, :]
        d_state, logit, aux_loss = cycle_model(current_check, d_state)
        logits_over_time.append(logit)
        aux_loss_total = aux_loss_total + aux_loss

    return jnp.stack(logits_over_time, axis=1), aux_loss_total


network = hk.transform(unroll_model)


def load_checkpoint(path):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    print(f"  Checkpoint loaded from {path}")
    return checkpoint["params"], checkpoint["opt_state"], checkpoint["epoch"]


@jax.jit
def predict_batch(params, rng, syndromes):
    """Run forward pass, return sigmoid probabilities for the final round."""
    logits, _ = network.apply(params, rng, syndromes)
    probs = jax.nn.sigmoid(logits)
    return probs


# --- 2. FORMAT TEST DATA ---
def format_test_data(det_events, measurements, obs_flips, rounds, batch_size):
    """
    Convert raw arrays into the model's standard 4-channel input format.
    Same convention as HardwareDataLoader._format_batch.
    """
    limit = rounds * NUM_STABILIZERS

    p1 = det_events[:, :limit].reshape(batch_size, rounds, NUM_STABILIZERS)
    meas_reshaped = measurements.reshape(batch_size, rounds + 1, NUM_STABILIZERS)
    e1 = meas_reshaped[:, 1:, :]
    p2 = np.zeros_like(p1)
    e2 = np.zeros_like(p1)

    syndromes = np.stack([p1, e1, p2, e2], axis=-1)
    labels = np.broadcast_to(
        obs_flips.reshape(batch_size, 1, 1),
        (batch_size, rounds, 1),
    ).copy().astype(np.float32)

    return jnp.array(syndromes), jnp.array(labels)


# --- 3. MAIN ---
def main():
    print("=" * 60)
    print("AlphaQubit Test Evaluation on Experimental Data")
    print(f"  Test round count: {TEST_ROUND}")
    print(f"  Code distance: {DISTANCE}")
    print("=" * 60)

    # Load fine-tuned checkpoint (prefer finetuned_best, fall back to best)
    ft_best = os.path.join(CHECKPOINT_DIR, "finetuned_best.pkl")
    ft_latest = os.path.join(CHECKPOINT_DIR, "finetuned_latest.pkl")
    pretrained = os.path.join(CHECKPOINT_DIR, "best.pkl")

    ckpt_path = None
    for candidate in [ft_best, ft_latest, pretrained]:
        if os.path.exists(candidate):
            ckpt_path = candidate
            break

    if ckpt_path is None:
        print("ERROR: No checkpoint found. Run train.py or finetune.py first.")
        return

    print(f"\nLoading checkpoint: {ckpt_path}")
    params, _, _ = load_checkpoint(ckpt_path)

    # Load test data
    print(f"\nLoading test data (r={TEST_ROUND:02d})...")
    det_events, measurements, obs_flips = load_hardware_data(
        data_dir=config.data_dir,
        rounds=TEST_ROUND,
        d=DISTANCE,
    )
    num_shots = det_events.shape[0]
    print(f"  {num_shots} test shots loaded")

    # Run model predictions in batches
    print("\nRunning model inference...")
    rng = jax.random.PRNGKey(0)
    all_final_preds = []

    for start in range(0, num_shots, EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, num_shots)
        batch_det = det_events[start:end]
        batch_meas = measurements[start:end]
        batch_obs = obs_flips[start:end]
        batch_size = end - start

        syndromes, targets = format_test_data(
            batch_det, batch_meas, batch_obs, TEST_ROUND, batch_size
        )

        rng, pred_rng = jax.random.split(rng)
        probs = predict_batch(params, pred_rng, syndromes)

        # Extract final-round prediction: (batch, 1)
        final_probs = np.array(probs[:, -1, :])
        final_preds = (final_probs > 0.5).astype(np.float32).flatten()
        all_final_preds.append(final_preds)

        if (start // EVAL_BATCH_SIZE) % 20 == 0:
            print(f"  Processed {end}/{num_shots} shots...")

    all_final_preds = np.concatenate(all_final_preds)

    # Compute AlphaQubit logical error rate
    ground_truth = obs_flips.flatten()
    aq_errors = np.sum(all_final_preds != ground_truth)
    aq_ler = aq_errors / num_shots

    # Load baseline decoder predictions
    print(f"\nLoading baseline decoder predictions...")
    baselines = load_baseline_predictions(
        data_dir=config.data_dir,
        rounds=TEST_ROUND,
        d=DISTANCE,
    )

    # Compute LER for each baseline
    baseline_results = {}
    for name, preds in baselines.items():
        errors = np.sum(preds.flatten() != ground_truth)
        ler = errors / num_shots
        baseline_results[name] = {"errors": int(errors), "ler": ler}

    # --- Print Results ---
    print("\n" + "=" * 60)
    print(f"RESULTS: Surface Code d={DISTANCE}, r={TEST_ROUND}, {num_shots} shots")
    print("=" * 60)
    print(f"{'Decoder':<40s} {'Errors':>8s} {'LER':>10s}")
    print("-" * 60)

    # AlphaQubit result
    print(f"{'AlphaQubit (ours)':<40s} {aq_errors:>8d} {aq_ler:>10.4f}")

    # Baselines sorted by LER
    for name in sorted(baseline_results, key=lambda k: baseline_results[k]["ler"]):
        r = baseline_results[name]
        display_name = name.replace("_", " ").title()
        print(f"{display_name:<40s} {r['errors']:>8d} {r['ler']:>10.4f}")

    print("-" * 60)

    # Relative improvement over best baseline
    if baseline_results:
        best_baseline_name = min(baseline_results, key=lambda k: baseline_results[k]["ler"])
        best_baseline_ler = baseline_results[best_baseline_name]["ler"]
        if best_baseline_ler > 0:
            relative_improvement = (best_baseline_ler - aq_ler) / best_baseline_ler * 100
            better_or_worse = "better" if relative_improvement > 0 else "worse"
            print(
                f"\nAlphaQubit is {abs(relative_improvement):.1f}% {better_or_worse} "
                f"than best baseline ({best_baseline_name.replace('_', ' ').title()})"
            )
        elif aq_ler == 0:
            print("\nBoth AlphaQubit and best baseline achieved 0 errors.")
        else:
            print(f"\nBest baseline has 0 errors; AlphaQubit has {aq_errors} errors.")

    print()


if __name__ == "__main__":
    main()
