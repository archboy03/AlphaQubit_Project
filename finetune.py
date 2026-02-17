import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import pickle
import os
import subprocess
import atexit
import signal
from datetime import datetime
from tensorboardX import SummaryWriter
from model.model import CycleArchitecture
from data_utils.hardware_data_loader import HardwareDataLoader

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

BATCH_SIZE = config.finetune_batch_size
LEARNING_RATE = config.finetune_lr
WEIGHT_DECAY = config.finetune_weight_decay
TOTAL_STEPS = config.finetune_steps
EVAL_EVERY = config.finetune_eval_every
LOG_EVERY = config.finetune_log_every
CHECKPOINT_EVERY = config.finetune_checkpoint_every
EARLY_STOPPING_PATIENCE = config.early_stopping_patience
CHECKPOINT_DIR = config.checkpoint_dir
PRETRAINED_CKPT = config.pretrained_checkpoint
FINETUNE_ROUNDS = config.finetune_rounds

# MoE load-balancing auxiliary loss coefficient
AUX_LOSS_COEF = 0.01

# Maximum number of shots per eval batch (to avoid OOM)
EVAL_BATCH_SIZE = 256

# --- 1. DATA LOADERS ---
print("=" * 60)
print("AlphaQubit Fine-tuning on Experimental Hardware Data")
print("=" * 60)

train_loader = HardwareDataLoader(config, split="train")
val_loader = HardwareDataLoader(config, split="val")


# --- 2. MODEL (identical to train.py) ---
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

# Optimizer: AdamW (weight decay for regularization) + gradient clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
)


# --- 3. LOSS & UPDATE FUNCTIONS ---
def loss_fn(params, rng, syndromes, targets):
    logits, aux_loss = network.apply(params, rng, syndromes)
    task_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets))
    return task_loss + AUX_LOSS_COEF * aux_loss


@jax.jit
def update_step(params, opt_state, rng, syndromes, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, rng, syndromes, targets)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


@jax.jit
def eval_step(params, rng, syndromes, targets):
    """Compute loss and accuracy on a batch without gradient updates."""
    logits, aux_loss = network.apply(params, rng, syndromes)
    task_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets))
    loss = task_loss + AUX_LOSS_COEF * aux_loss
    predictions = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    mean_accuracy = jnp.mean(predictions == targets)
    final_accuracy = jnp.mean(predictions[:, -1, :] == targets[:, -1, :])
    return loss, mean_accuracy, final_accuracy


# --- 4. CHECKPOINTING ---
def save_checkpoint(params, opt_state, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {"params": params, "opt_state": opt_state, "epoch": step}
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  Checkpoint saved to {path}")


def load_checkpoint(path):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    print(f"  Checkpoint loaded from {path}")
    return checkpoint["params"], checkpoint["opt_state"], checkpoint["epoch"]


# --- 5. ROUND-COUNT CURRICULUM ---
def sample_round_count(step, total_steps, round_counts):
    """
    Sample a round count with curriculum: early training biases toward
    low rounds, gradually becoming uniform over all rounds.

    The curriculum works by restricting which rounds are available.
    At progress=0, only the lowest round is used.
    At progress=1, all rounds are available uniformly.
    """
    progress = min(max(step / total_steps, 0.0), 1.0) if total_steps > 0 else 1.0
    sorted_rounds = sorted(round_counts)
    num_available = max(1, int(np.ceil(len(sorted_rounds) * progress)))
    available = sorted_rounds[:num_available]
    return int(np.random.choice(available))


# --- 6. VALIDATION ---
def run_validation(params, rng, val_loader, round_counts):
    """
    Evaluate on the validation split across all round counts.
    Returns per-round metrics and aggregate metrics.
    """
    total_loss = 0.0
    total_final_correct = 0
    total_final_shots = 0
    per_round_results = {}

    for r in round_counts:
        n = val_loader.num_shots(r)
        if n == 0:
            continue

        # Evaluate in chunks to avoid OOM
        round_losses = []
        round_final_correct = 0
        round_total = 0

        for start in range(0, n, EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, n)
            chunk_size = end - start
            syndromes, targets = val_loader.get_batch(chunk_size, r)
            rng, eval_rng = jax.random.split(rng)
            loss, mean_acc, final_acc = eval_step(params, eval_rng, syndromes, targets)

            round_losses.append(float(loss) * chunk_size)
            round_final_correct += int(float(final_acc) * chunk_size)
            round_total += chunk_size

        avg_loss = sum(round_losses) / round_total
        final_acc = round_final_correct / round_total

        per_round_results[r] = {"loss": avg_loss, "final_accuracy": final_acc, "shots": round_total}
        total_loss += sum(round_losses)
        total_final_correct += round_final_correct
        total_final_shots += round_total

    agg_loss = total_loss / total_final_shots if total_final_shots > 0 else 0.0
    agg_final_acc = total_final_correct / total_final_shots if total_final_shots > 0 else 0.0

    return agg_loss, agg_final_acc, per_round_results, rng


# --- 7. INITIALIZATION ---
rng = jax.random.PRNGKey(42)
start_step = 0

# Try to resume from a fine-tuning checkpoint first
ft_latest_ckpt = os.path.join(CHECKPOINT_DIR, "finetuned_latest.pkl")
if os.path.exists(ft_latest_ckpt):
    print("\nResuming fine-tuning from checkpoint...")
    params, opt_state, start_step = load_checkpoint(ft_latest_ckpt)
    print(f"  Resuming from step {start_step}")
    for _ in range(start_step + 1):
        rng, _ = jax.random.split(rng)
elif os.path.exists(PRETRAINED_CKPT):
    print(f"\nLoading pre-trained weights from {PRETRAINED_CKPT}...")
    params, _, _ = load_checkpoint(PRETRAINED_CKPT)
    # Fresh optimizer state for fine-tuning
    opt_state = optimizer.init(params)
    print("  Pre-trained weights loaded. Optimizer state initialized fresh.")
else:
    print("\nNo pre-trained checkpoint found. Initializing from scratch...")
    # Need a dummy input to initialize the model -- use the smallest round count
    min_r = min(FINETUNE_ROUNDS)
    dummy_syndromes, _ = train_loader.get_batch(2, min_r)
    params = network.init(rng, dummy_syndromes)
    opt_state = optimizer.init(params)

print(f"\nFine-tuning configuration:")
print(f"  distance={DISTANCE}, stabilizers={NUM_STABILIZERS}")
print(f"  hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}")
print(f"  learning_rate={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, batch_size={BATCH_SIZE}")
print(f"  total_steps={TOTAL_STEPS}, early_stopping_patience={EARLY_STOPPING_PATIENCE}")
print(f"  round_counts={FINETUNE_ROUNDS}")

# --- 8. TENSORBOARD ---
run_name = "finetune_" + datetime.now().strftime("%Y%m%d_%H%M%S")
run_log_dir = os.path.join(config.tensorboard_dir, run_name)
writer = SummaryWriter(run_log_dir)

TB_PORT = 6007
tb_process = subprocess.Popen(
    ["tensorboard", "--logdir", config.tensorboard_dir, "--port", str(TB_PORT)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)


def _cleanup_tensorboard():
    tb_process.terminate()
    tb_process.wait()


atexit.register(_cleanup_tensorboard)
signal.signal(signal.SIGTERM, lambda sig, frame: (_cleanup_tensorboard(), exit(0)))

print(f"TensorBoard started on: \033[4mhttp://localhost:{TB_PORT}\033[0m\n")

# --- 9. FINE-TUNING LOOP ---
print("Starting fine-tuning...")
best_eval_loss = float("inf")
evals_without_improvement = 0
stopped_early = False

try:
    for step in range(start_step, TOTAL_STEPS):
        # 1. Sample round count with curriculum
        r = sample_round_count(step, TOTAL_STEPS, FINETUNE_ROUNDS)

        # 2. Get a training batch
        syndromes, targets = train_loader.get_batch(BATCH_SIZE, r)

        # 3. Update weights
        rng, step_rng = jax.random.split(rng)
        params, opt_state, loss_val = update_step(
            params, opt_state, step_rng, syndromes, targets
        )

        # 4. Logging
        if step % LOG_EVERY == 0:
            loss_scalar = float(loss_val)
            print(f"Step {step}: Loss = {loss_scalar:.4f} (r={r})")
            writer.add_scalar("finetune/loss", loss_scalar, step)
            writer.add_scalar("finetune/round_count", r, step)
            writer.flush()

        # 5. Validation + early stopping
        if (step + 1) % EVAL_EVERY == 0:
            agg_loss, agg_final_acc, per_round, rng = run_validation(
                params, rng, val_loader, FINETUNE_ROUNDS
            )
            print(f"  [EVAL] Step {step}: Loss = {agg_loss:.4f}, Final Acc = {agg_final_acc:.4f}")

            writer.add_scalar("eval/loss", agg_loss, step)
            writer.add_scalar("eval/final_accuracy", agg_final_acc, step)
            for r_val, metrics in per_round.items():
                writer.add_scalar(f"eval/final_acc_r{r_val:02d}", metrics["final_accuracy"], step)
            writer.flush()

            # Save best model + track early stopping
            if agg_loss < best_eval_loss:
                best_eval_loss = agg_loss
                evals_without_improvement = 0
                save_checkpoint(
                    params, opt_state, step,
                    os.path.join(CHECKPOINT_DIR, "finetuned_best.pkl"),
                )
            else:
                evals_without_improvement += 1
                print(f"  No improvement for {evals_without_improvement}/{EARLY_STOPPING_PATIENCE} evals")

            if evals_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping triggered after {EARLY_STOPPING_PATIENCE} evals without improvement.")
                stopped_early = True
                save_checkpoint(params, opt_state, step, ft_latest_ckpt)
                break

        # 6. Periodic checkpointing
        if (step + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, opt_state, step, ft_latest_ckpt)

except KeyboardInterrupt:
    print("\nFine-tuning stopped manually.")
    save_checkpoint(params, opt_state, step, ft_latest_ckpt)
else:
    if not stopped_early:
        # Save final checkpoint when training completed all steps normally
        save_checkpoint(params, opt_state, TOTAL_STEPS - 1, ft_latest_ckpt)

writer.close()
reason = "early stopping" if stopped_early else "all steps completed"
print(f"\nFine-tuning complete ({reason}).")
print(f"  Best eval loss: {best_eval_loss:.4f}")
print(f"  Best checkpoint: {os.path.join(CHECKPOINT_DIR, 'finetuned_best.pkl')}")
print(f"TensorBoard still running at: \033[4mhttp://localhost:{TB_PORT}\033[0m")
print("Press Ctrl+C to stop.")
try:
    tb_process.wait()
except KeyboardInterrupt:
    pass
