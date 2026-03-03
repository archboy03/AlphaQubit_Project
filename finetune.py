"""
Finetuning script for the AlphaQubit model on experimental Sycamore chip data.

Mirrors the structure of train.py, but loads pre-recorded experimental data
from disk instead of generating synthetic data online. Iterates through
round folders (r01 → r03 → ... → r25) in a curriculum-like progression.

Usage:
    python finetune.py
"""

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pickle
import os
import subprocess
import atexit
import signal
import shutil
from datetime import datetime
from tensorboardX import SummaryWriter
from model.model import CycleArchitecture
from data_utils.experimental_data_loader import ExperimentalDataLoader, discover_round_folders

try:
    from config_local import LocalConfig as Config
except ImportError:
    from config import Config

config = Config()

# --- CONFIGURATION ---
BATCH_SIZE = config.batch_size
DISTANCE = config.code_distance
LEARNING_RATE = config.finetune_learning_rate
EPOCHS_PER_ROUND = config.finetune_epochs
EVAL_EVERY = 50
EVAL_SHOTS = 256
CHECKPOINT_DIR = config.finetune_checkpoint_dir
CHECKPOINT_EVERY = config.checkpoint_every
LOG_EVERY = config.log_every
HIDDEN_SIZE = config.hidden_size
NUM_LAYERS = config.num_layers
POSITIVE_CLASS_WEIGHT = config.positive_class_weight
FINAL_ROUND_WEIGHT = config.final_round_loss_weight
PRETRAINED_CKPT = config.pretrained_checkpoint

INPUT_FEATURES = 4
NUM_STABILIZERS = DISTANCE ** 2 - 1  # 8 for d=3

# --- 1. MODEL DEFINITION ---
# NOTE: Because the Python for-loop range depends on syndromes.shape[1],
# JAX will re-trace (recompile) once per unique round count. This is expected.

def unroll_model(syndromes):
    """
    syndromes: (Batch, Rounds+1, NUM_STABILIZERS, 4)
    Returns:   (Batch, Rounds+1, 1)
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

    for t in range(syndromes.shape[1]):
        current_check = syndromes[:, t, :, :]
        d_state, logit = cycle_model(current_check, d_state)
        logits_over_time.append(logit)

    return jnp.stack(logits_over_time, axis=1)


network = hk.transform(unroll_model)

# Optimizer: lower LR + gradient clipping for stable finetuning
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(LEARNING_RATE),
)

# --- 2. LOSS & UPDATE FUNCTIONS ---
def loss_fn(params, rng, syndromes, targets):
    logits = network.apply(params, rng, syndromes)
    raw_bce = optax.sigmoid_binary_cross_entropy(logits, targets)

    # Class weighting: up-weight the positive (error) class to combat imbalance
    class_weights = jnp.where(targets == 1.0, POSITIVE_CLASS_WEIGHT, 1.0)

    # Final-round weighting: emphasize the last round (the actual decoding target)
    round_weights = jnp.ones_like(raw_bce)
    round_weights = round_weights.at[:, -1, :].set(FINAL_ROUND_WEIGHT)

    loss = jnp.mean(raw_bce * class_weights * round_weights)
    return loss


@jax.jit
def update_step(params, opt_state, rng, syndromes, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, rng, syndromes, targets)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


@jax.jit
def eval_step(params, rng, syndromes, targets):
    """Compute loss and accuracy on a batch without gradient updates."""
    logits = network.apply(params, rng, syndromes)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets))
    probs = jax.nn.sigmoid(logits)
    predictions = (probs > 0.5).astype(jnp.float32)
    mean_accuracy = jnp.mean(predictions == targets)
    final_accuracy = jnp.mean(predictions[:, -1, :] == targets[:, -1, :])
    prob_mean = jnp.mean(probs)
    prob_std = jnp.std(probs)
    frac_positive = jnp.mean(predictions)
    return loss, mean_accuracy, final_accuracy, prob_mean, prob_std, frac_positive


# --- 3. CHECKPOINTING ---
def save_checkpoint(params, opt_state, epoch, path):
    """Save model parameters and optimizer state to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "params": params,
        "opt_state": opt_state,
        "epoch": epoch,
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  Checkpoint saved to {path}")


def load_checkpoint(path):
    """Load model parameters and optimizer state from disk."""
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    print(f"  Checkpoint loaded from {path}")
    return checkpoint["params"], checkpoint["opt_state"], checkpoint["epoch"]


# --- 4. LOAD PRETRAINED WEIGHTS ---
assert os.path.exists(PRETRAINED_CKPT), (
    f"Pretrained checkpoint not found: {PRETRAINED_CKPT}\n"
    f"Please train the model first using train.py"
)

print(f"Loading pretrained weights from: {PRETRAINED_CKPT}")
params, _, _ = load_checkpoint(PRETRAINED_CKPT)

# Fresh optimizer state for the new (lower) learning rate
opt_state = optimizer.init(params)
print(f"  Optimizer re-initialized with LR={LEARNING_RATE}")

rng = jax.random.PRNGKey(123)

# --- 5. DISCOVER AVAILABLE ROUNDS ---
available_rounds = discover_round_folders(config)
assert len(available_rounds) > 0, (
    f"No experimental data folders found in: {config.data_dir}"
)
print(f"Found experimental data for {len(available_rounds)} round counts: {available_rounds}")

# --- 6. TENSORBOARD ---
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.finetune_tensorboard_dir, exist_ok=True)

if getattr(config, "tensorboard_latest_only", False):
    run_log_dir = os.path.join(config.finetune_tensorboard_dir, "latest")
    if os.path.exists(run_log_dir):
        shutil.rmtree(run_log_dir)
    os.makedirs(run_log_dir, exist_ok=True)
    tb_logdir = run_log_dir
else:
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(config.finetune_tensorboard_dir, run_name)
    tb_logdir = config.finetune_tensorboard_dir

try:
    writer = SummaryWriter(run_log_dir)
    TB_PORT = 6007  # Use different port from train.py (6006)
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir", tb_logdir, "--port", str(TB_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _cleanup_tensorboard():
        tb_process.terminate()
        tb_process.wait()

    atexit.register(_cleanup_tensorboard)
    signal.signal(signal.SIGTERM, lambda sig, frame: (_cleanup_tensorboard(), exit(0)))
    print(f"TensorBoard started on: \033[4mhttp://localhost:{TB_PORT}\033[0m")
except OSError as e:
    class _NoOpWriter:
        def add_scalar(self, *args, **kwargs): pass
        def flush(self): pass
        def close(self): pass
    writer = _NoOpWriter()
    tb_process = None
    print(f"TensorBoard disabled (could not create log dir: {e})")

# --- 7. FINETUNING LOOP ---
print("\n" + "=" * 60)
print("Starting finetuning on experimental data...")
print("=" * 60)

best_eval_loss = float("inf")
global_step = 0

try:
    for round_idx, num_rounds in enumerate(available_rounds):
        print(f"\n--- Round folder {round_idx + 1}/{len(available_rounds)}: "
              f"r={num_rounds:02d} ---")

        # Load experimental data for this round count
        data_loader = ExperimentalDataLoader(config, rounds=num_rounds)

        for epoch in range(EPOCHS_PER_ROUND):
            # 1. Sample a batch from experimental data
            syndromes, targets = data_loader.get_batch(BATCH_SIZE)

            rng, step_rng = jax.random.split(rng)

            # 2. Update weights
            params, opt_state, loss_val = update_step(
                params, opt_state, step_rng, syndromes, targets
            )

            if epoch % LOG_EVERY == 0:
                loss_scalar = float(loss_val)
                print(f"  [r={num_rounds:02d}] Epoch {epoch}: Loss = {loss_scalar:.4f}")
                writer.add_scalar("finetune/train_loss", loss_scalar, global_step)
                writer.add_scalar("finetune/round", num_rounds, global_step)
                writer.flush()

            # 3. Evaluation
            if (epoch + 1) % EVAL_EVERY == 0:
                rng, eval_rng = jax.random.split(rng)
                eval_syndromes, eval_targets = data_loader.get_eval_batch(
                    min(EVAL_SHOTS, BATCH_SIZE)
                )
                eval_loss, eval_mean_acc, eval_final_acc, pred_mean, pred_std, frac_pos = eval_step(
                    params, eval_rng, eval_syndromes, eval_targets
                )
                eval_loss_scalar = float(eval_loss)
                eval_mean_scalar = float(eval_mean_acc)
                eval_final_scalar = float(eval_final_acc)
                print(
                    f"  [EVAL r={num_rounds:02d}] Epoch {epoch}: "
                    f"Loss = {eval_loss_scalar:.4f}, "
                    f"Mean Acc = {eval_mean_scalar:.4f}, "
                    f"Final Acc = {eval_final_scalar:.4f}, "
                    f"Pred μ={float(pred_mean):.4f} σ={float(pred_std):.4f} pos={float(frac_pos):.3f}"
                )
                writer.add_scalar("finetune/eval_loss", eval_loss_scalar, global_step)
                writer.add_scalar("finetune/eval_mean_accuracy", eval_mean_scalar, global_step)
                writer.add_scalar("finetune/eval_final_accuracy", eval_final_scalar, global_step)
                writer.add_scalar("finetune/pred_prob_mean", float(pred_mean), global_step)
                writer.add_scalar("finetune/pred_prob_std", float(pred_std), global_step)
                writer.add_scalar("finetune/pred_frac_positive", float(frac_pos), global_step)
                writer.flush()

                # Save best model
                if eval_loss_scalar < best_eval_loss:
                    best_eval_loss = eval_loss_scalar
                    save_checkpoint(
                        params, opt_state, global_step,
                        os.path.join(CHECKPOINT_DIR, "best.pkl"),
                    )

            # 4. Periodic checkpointing
            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(
                    params, opt_state, global_step,
                    os.path.join(CHECKPOINT_DIR, "latest.pkl"),
                )

            global_step += 1

        # Save checkpoint after finishing each round folder
        save_checkpoint(
            params, opt_state, global_step,
            os.path.join(CHECKPOINT_DIR, f"after_r{num_rounds:02d}.pkl"),
        )

except KeyboardInterrupt:
    print("\nFinetuning stopped manually.")
    save_checkpoint(
        params, opt_state, global_step,
        os.path.join(CHECKPOINT_DIR, "latest.pkl"),
    )

writer.close()
print("\nFinetuning complete.")
if tb_process is not None:
    print(f"TensorBoard still running at: \033[4mhttp://localhost:{TB_PORT}\033[0m")
    print("Press Ctrl+C to stop.")
    try:
        tb_process.wait()
    except KeyboardInterrupt:
        pass
