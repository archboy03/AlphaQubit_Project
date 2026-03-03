"""
Final evaluation script for the AlphaQubit model on experimental data.

Evaluates the model on the full held-out (20%) eval set for every round count,
reporting definitive metrics: loss, mean accuracy, final-round accuracy,
and logical error rate (LER). Optionally compares against baseline decoders
(pymatching, belief matching, etc.) from the experimental .01 files.

Usage:
    python test.py                                         # Evaluate finetuned best checkpoint
    python test.py --checkpoint checkpoints/latest.pkl     # Evaluate a specific checkpoint
    python test.py --checkpoint checkpoints/finetune/best.pkl
"""

import argparse
import os
import math
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pickle

from model.model import CycleArchitecture
from data_utils.experimental_data_loader import ExperimentalDataLoader, discover_round_folders

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
EVAL_BATCH_SIZE = 256  # Process eval data in chunks to avoid OOM


# --- 1. MODEL DEFINITION (must match train.py / finetune.py) ---
def unroll_model(syndromes):
    cycle_model = CycleArchitecture(
        mixing_mult=0.5,
        output_size=HIDDEN_SIZE,
        distance=DISTANCE,
        num_layers=NUM_LAYERS,
    )
    batch_size = syndromes.shape[0]
    d_state = jnp.zeros((batch_size, NUM_STABILIZERS, HIDDEN_SIZE))
    logits_over_time = []
    for t in range(syndromes.shape[1]):
        current_check = syndromes[:, t, :, :]
        d_state, logit = cycle_model(current_check, d_state)
        logits_over_time.append(logit)
    return jnp.stack(logits_over_time, axis=1)


network = hk.transform(unroll_model)


@jax.jit
def predict_batch(params, rng, syndromes):
    """Run inference and return logits."""
    return network.apply(params, rng, syndromes)


# --- 2. BASELINE DECODER COMPARISON ---
def load_baseline_results(data_dir, d, rounds):
    """Load decoder predictions from .01 files for comparison.

    Returns dict of decoder_name -> error_rate (float).
    """
    folder_name = f"surface_code_bZ_d{d}_r{rounds:02d}_center_3_5"
    folder_path = os.path.join(data_dir, folder_name)

    baselines = {}
    decoder_files = {
        "PyMatching": "obs_flips_predicted_by_pymatching.01",
        "Belief Matching": "obs_flips_predicted_by_belief_matching.01",
        "Correlated Matching": "obs_flips_predicted_by_correlated_matching.01",
        "Tensor Network": "obs_flips_predicted_by_tensor_network_contraction.01",
    }

    # Load actual labels
    actual_path = os.path.join(folder_path, "obs_flips_actual.01")
    if not os.path.exists(actual_path):
        return baselines
    with open(actual_path, "r") as f:
        actual = np.array([int(ch) for ch in f.read() if ch in ("0", "1")], dtype=np.int32)

    for name, filename in decoder_files.items():
        pred_path = os.path.join(folder_path, filename)
        if os.path.exists(pred_path):
            with open(pred_path, "r") as f:
                preds = np.array([int(ch) for ch in f.read() if ch in ("0", "1")], dtype=np.int32)
            # These predictions are whether the decoder thinks the observable flipped.
            # A "logical error" is when the decoder's prediction disagrees with reality.
            logical_errors = (preds != actual).astype(np.float32)
            baselines[name] = float(logical_errors.mean())

    return baselines


# --- 3. MAIN EVALUATION ---
def evaluate_all_rounds(params, rng, config, available_rounds):
    """Evaluate the model on the full eval set for every round count."""

    all_results = []

    for num_rounds in available_rounds:
        print(f"\n--- Evaluating r={num_rounds:02d} ---")

        # Load all eval data for this round
        loader = ExperimentalDataLoader(config, rounds=num_rounds)
        n_eval = loader.eval_syndromes.shape[0]

        # Evaluate in batches
        all_logits = []
        all_targets = []

        num_batches = math.ceil(n_eval / EVAL_BATCH_SIZE)
        for i in range(num_batches):
            start = i * EVAL_BATCH_SIZE
            end = min(start + EVAL_BATCH_SIZE, n_eval)
            batch_syn = jnp.array(loader.eval_syndromes[start:end])
            batch_tgt = jnp.array(loader.eval_targets[start:end])

            rng, step_rng = jax.random.split(rng)
            logits = predict_batch(params, step_rng, batch_syn)

            all_logits.append(logits)
            all_targets.append(batch_tgt)

        # Concatenate all batches
        all_logits = jnp.concatenate(all_logits, axis=0)
        all_targets = jnp.concatenate(all_targets, axis=0)

        # --- Compute metrics ---
        # Loss
        loss = float(jnp.mean(optax.sigmoid_binary_cross_entropy(all_logits, all_targets)))

        # Predictions
        probs = jax.nn.sigmoid(all_logits)
        preds = (probs > 0.5).astype(jnp.float32)

        # Mean accuracy (across all rounds)
        mean_acc = float(jnp.mean(preds == all_targets))

        # Final-round accuracy
        final_preds = preds[:, -1, :]
        final_targets = all_targets[:, -1, :]
        final_acc = float(jnp.mean(final_preds == final_targets))

        # Logical Error Rate (LER) = fraction of shots where final prediction is wrong
        ler = 1.0 - final_acc

        result = {
            "rounds": num_rounds,
            "n_eval": n_eval,
            "loss": loss,
            "mean_accuracy": mean_acc,
            "final_accuracy": final_acc,
            "logical_error_rate": ler,
        }

        # Load baseline decoders for comparison
        baselines = load_baseline_results(config.data_dir, config.code_distance, num_rounds)
        result["baselines"] = baselines

        all_results.append(result)

        # Print per-round results
        print(f"  Eval shots:   {n_eval}")
        print(f"  Loss:         {loss:.6f}")
        print(f"  Mean Acc:     {mean_acc:.4f}")
        print(f"  Final Acc:    {final_acc:.4f}")
        print(f"  LER:          {ler:.6f}")
        if baselines:
            print(f"  --- Baseline Decoders ---")
            for decoder_name, baseline_ler in baselines.items():
                diff = ler - baseline_ler
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"    {decoder_name:25s}: LER = {baseline_ler:.6f}  (diff: {diff:+.6f} {symbol})")

    return all_results


def print_summary_table(results):
    """Print a summary table of all results."""
    print("\n" + "=" * 90)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 90)

    # Header
    header = f"{'Rounds':>6} | {'Shots':>6} | {'Loss':>10} | {'Mean Acc':>10} | {'Final Acc':>10} | {'LER':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['rounds']:>6d} | {r['n_eval']:>6d} | {r['loss']:>10.6f} | "
            f"{r['mean_accuracy']:>10.4f} | {r['final_accuracy']:>10.4f} | "
            f"{r['logical_error_rate']:>10.6f}"
        )

    # Overall averages
    avg_loss = np.mean([r["loss"] for r in results])
    avg_mean_acc = np.mean([r["mean_accuracy"] for r in results])
    avg_final_acc = np.mean([r["final_accuracy"] for r in results])
    avg_ler = np.mean([r["logical_error_rate"] for r in results])
    print("-" * len(header))
    print(
        f"{'AVG':>6s} | {'':>6s} | {avg_loss:>10.6f} | "
        f"{avg_mean_acc:>10.4f} | {avg_final_acc:>10.4f} | "
        f"{avg_ler:>10.6f}"
    )

    # Baseline comparison table (if available)
    any_baselines = any(r.get("baselines") for r in results)
    if any_baselines:
        print("\n" + "=" * 90)
        print("LOGICAL ERROR RATE COMPARISON vs BASELINE DECODERS")
        print("=" * 90)

        # Collect all decoder names
        all_decoders = set()
        for r in results:
            all_decoders.update(r.get("baselines", {}).keys())
        all_decoders = sorted(all_decoders)

        header2 = f"{'Rounds':>6} | {'AlphaQubit':>12}"
        for name in all_decoders:
            header2 += f" | {name:>15}"
        print(header2)
        print("-" * len(header2))

        for r in results:
            line = f"{r['rounds']:>6d} | {r['logical_error_rate']:>12.6f}"
            for name in all_decoders:
                if name in r.get("baselines", {}):
                    line += f" | {r['baselines'][name]:>15.6f}"
                else:
                    line += f" | {'N/A':>15s}"
            print(line)

    print("\n" + "=" * 90)


# --- 4. ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AlphaQubit on experimental data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.finetune_checkpoint_dir, "best.pkl"),
        help="Path to the checkpoint to evaluate (default: finetune best.pkl)",
    )
    args = parser.parse_args()

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        # Fallback: try latest pretrained
        fallbacks = [
            os.path.join(config.finetune_checkpoint_dir, "latest.pkl"),
            os.path.join(config.checkpoint_dir, "best.pkl"),
            os.path.join(config.checkpoint_dir, "latest.pkl"),
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                ckpt_path = fb
                print(f"Requested checkpoint not found, falling back to: {fb}")
                break
        else:
            print(f"ERROR: No checkpoint found. Tried:\n  {args.checkpoint}")
            for fb in fallbacks:
                print(f"  {fb}")
            exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)
    params = checkpoint["params"]
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")

    # Discover rounds
    available_rounds = discover_round_folders(config)
    print(f"Found {len(available_rounds)} round folders: {available_rounds}")

    # Run evaluation
    rng = jax.random.PRNGKey(0)
    results = evaluate_all_rounds(params, rng, config, available_rounds)

    # Print summary
    print_summary_table(results)
