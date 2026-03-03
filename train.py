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
from data_utils.data_loader import CurriculumDataLoader

try:
    from config_local import LocalConfig as Config
except ImportError:
    from config import Config

config = Config()

# --- CONFIGURATION (from config.py) ---
BATCH_SIZE = config.batch_size
ROUNDS = config.rounds
DISTANCE = config.code_distance
LEARNING_RATE = config.learning_rate
TOTAL_EPOCHS = config.num_steps
EVAL_EVERY = 50
EVAL_SHOTS = 256
CHECKPOINT_DIR = config.checkpoint_dir
CHECKPOINT_EVERY = config.checkpoint_every
LOG_EVERY = config.log_every
HIDDEN_SIZE = config.hidden_size
NUM_LAYERS = config.num_layers
POSITIVE_CLASS_WEIGHT = config.positive_class_weight
FINAL_ROUND_WEIGHT = config.final_round_loss_weight

# We need to know input features for the embedder
# Your data provides 4 channels: Post_1, Event_1, Post_2, Event_2
INPUT_FEATURES = 4

# Derived constants
NUM_STABILIZERS = DISTANCE**2 - 1  # 8 for d=3

# --- 1. DATA LOADER ---
data_loader = CurriculumDataLoader(config)

# --- 2. THE MODEL WRAPPER ---
def unroll_model(syndromes):
    """
    syndromes: (Batch, Rounds, NUM_STABILIZERS, 4)
    Returns: (Batch, Rounds, 1)
    """
    # Initialize the architecture
    cycle_model = CycleArchitecture(
        mixing_mult=0.5,
        output_size=HIDDEN_SIZE,
        distance=DISTANCE,
        num_layers=NUM_LAYERS,
    )
    
    # Initial Decoder State: (Batch, NUM_STABILIZERS, HIDDEN_SIZE)
    batch_size = syndromes.shape[0]
    d_state = jnp.zeros((batch_size, NUM_STABILIZERS, HIDDEN_SIZE))
    
    logits_over_time = []

    # Loop over time
    for t in range(syndromes.shape[1]):
        # Current input: (Batch, NUM_STABILIZERS, 4)
        current_check = syndromes[:, t, :, :]

        # Run RNN step
        d_state, logit = cycle_model(current_check, d_state)
        logits_over_time.append(logit)

    return jnp.stack(logits_over_time, axis=1)

# Create the pure functions
network = hk.transform(unroll_model)

# Optimizer with gradient clipping and LR warmup for transformer training stability
WARMUP_STEPS = 1000
lr_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=LEARNING_RATE,
    transition_steps=WARMUP_STEPS,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(lr_schedule),
)

# --- 3. LOSS & UPDATE FUNCTIONS ---
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
    logits = network.apply(params, rng, syndromes)  # (B, R, 1)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets))
    probs = jax.nn.sigmoid(logits)
    predictions = (probs > 0.5).astype(jnp.float32)
    mean_accuracy = jnp.mean(predictions == targets)
    final_accuracy = jnp.mean(predictions[:, -1, :] == targets[:, -1, :])
    # Prediction distribution stats for collapse detection
    prob_mean = jnp.mean(probs)
    prob_std = jnp.std(probs)
    frac_positive = jnp.mean(predictions)
    return loss, mean_accuracy, final_accuracy, prob_mean, prob_std, frac_positive

# --- 4. CHECKPOINTING ---
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

# --- 5. MAIN TRAINING LOOP ---

# Initialization
print("Generating initialization batch...")
sample_input, sample_target = data_loader.get_batch(
    BATCH_SIZE, epoch=0, total_epochs=TOTAL_EPOCHS
)

rng = jax.random.PRNGKey(42)
start_epoch = 0

# Try to resume from latest checkpoint
latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pkl")
if os.path.exists(latest_ckpt):
    print("Resuming from checkpoint...")
    params, opt_state, start_epoch = load_checkpoint(latest_ckpt)
    print(f"  Resuming from epoch {start_epoch}")
    # Advance the RNG to match where we left off
    for _ in range(start_epoch + 1):
        rng, _ = jax.random.split(rng)
else:
    print("Initializing parameters from scratch...")
    params = network.init(rng, sample_input)
    opt_state = optimizer.init(params)

print(f"Model initialized. Input shape: {sample_input.shape}")
print(f"  distance={DISTANCE}, stabilizers={NUM_STABILIZERS}, rounds={ROUNDS}")
print(f"  hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}, mode={config.mode}")

# Ensure output directories exist before TensorBoard/checkpointing
os.makedirs(config.tensorboard_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)

# TensorBoard writer — optionally restrict to a single latest run
if getattr(config, "tensorboard_latest_only", False):
    # Use a fixed "latest" directory and clear it each run
    run_log_dir = os.path.join(config.tensorboard_dir, "latest")
    if os.path.exists(run_log_dir):
        shutil.rmtree(run_log_dir)
    os.makedirs(run_log_dir, exist_ok=True)
    tb_logdir = run_log_dir
else:
    # Preserve historical runs using timestamped subdirectories
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(config.tensorboard_dir, run_name)
    tb_logdir = config.tensorboard_dir

try:
    writer = SummaryWriter(run_log_dir)
    TB_PORT = 6006
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
    # Fallback when log dir is unwritable (e.g. read-only filesystem)
    class _NoOpWriter:
        def add_scalar(self, *args, **kwargs): pass
        def flush(self): pass
        def close(self): pass
    writer = _NoOpWriter()
    tb_process = None
    print(f"TensorBoard disabled (could not create log dir: {e})")

# Training
print("Starting training...")
best_eval_loss = float("inf")

try:
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # 1. Generate Fresh Data (Online Training with Curriculum)
        syndromes, targets = data_loader.get_batch(
            BATCH_SIZE, epoch=epoch, total_epochs=TOTAL_EPOCHS
        )
        
        rng, step_rng = jax.random.split(rng)
        
        # 2. Update Weights
        params, opt_state, loss_val = update_step(
            params, 
            opt_state, 
            step_rng, 
            syndromes, 
            targets
        )
        
        if epoch % LOG_EVERY == 0:
            loss_scalar = float(loss_val)
            print(f"Epoch {epoch}: Loss = {loss_scalar:.4f}")
            writer.add_scalar("train/loss", loss_scalar, epoch)

            # Log curriculum progress
            progress = min(max(epoch / TOTAL_EPOCHS, 0.0), 1.0) if TOTAL_EPOCHS > 0 else 1.0
            _, curr_p = data_loader._get_difficulty_params(progress)
            writer.add_scalar("curriculum/physical_error_rate", curr_p, epoch)
            writer.flush()

        # 3. Evaluation on held-out data
        if (epoch + 1) % EVAL_EVERY == 0:
            rng, eval_rng = jax.random.split(rng)
            eval_syndromes, eval_targets = data_loader.get_batch(
                EVAL_SHOTS, epoch=epoch, total_epochs=TOTAL_EPOCHS
            )
            eval_loss, eval_mean_acc, eval_final_acc, pred_mean, pred_std, frac_pos = eval_step(
                params, eval_rng, eval_syndromes, eval_targets
            )
            eval_loss_scalar = float(eval_loss)
            eval_mean_scalar = float(eval_mean_acc)
            eval_final_scalar = float(eval_final_acc)
            print(
                f"  [EVAL] Epoch {epoch}: Loss = {eval_loss_scalar:.4f}, "
                f"Mean Acc = {eval_mean_scalar:.4f}, Final Acc = {eval_final_scalar:.4f}, "
                f"Pred μ={float(pred_mean):.4f} σ={float(pred_std):.4f} pos={float(frac_pos):.3f}"
            )
            writer.add_scalar("eval/loss", eval_loss_scalar, epoch)
            writer.add_scalar("eval/mean_accuracy", eval_mean_scalar, epoch)
            writer.add_scalar("eval/final_accuracy", eval_final_scalar, epoch)
            writer.add_scalar("eval/pred_prob_mean", float(pred_mean), epoch)
            writer.add_scalar("eval/pred_prob_std", float(pred_std), epoch)
            writer.add_scalar("eval/pred_frac_positive", float(frac_pos), epoch)
            writer.flush()
            
            # Save best model
            if eval_loss_scalar < best_eval_loss:
                best_eval_loss = eval_loss_scalar
                save_checkpoint(params, opt_state, epoch, os.path.join(CHECKPOINT_DIR, "best.pkl"))

        # 4. Periodic checkpointing
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, opt_state, epoch, latest_ckpt)

except KeyboardInterrupt:
    print("\nTraining stopped manually.")
    save_checkpoint(params, opt_state, epoch, latest_ckpt)

writer.close()
print("Done.")
if tb_process is not None:
    print(f"TensorBoard still running at: \033[4mhttp://localhost:{TB_PORT}\033[0m")
    print("Press Ctrl+C to stop.")
    try:
        tb_process.wait()
    except KeyboardInterrupt:
        pass
