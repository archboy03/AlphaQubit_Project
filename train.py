import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import pickle
import os
from model.model import CycleArchitecture  # Your architecture file
from data_utils.generate_data import generate_training_data # Your data file

# --- CONFIGURATION ---
BATCH_SIZE = 32
ROUNDS = 5        # Number of QEC rounds
DISTANCE = 3      # Surface code distance
LEARNING_RATE = 1e-4
TOTAL_EPOCHS = 1000
EVAL_EVERY = 50           # Evaluate on held-out data every N epochs
EVAL_SHOTS = 256          # Number of shots for evaluation
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 100    # Save checkpoint every N epochs

# We need to know input features for the embedder
# Your data provides 4 channels: Post_1, Event_1, Post_2, Event_2
INPUT_FEATURES = 4 

# Derived constants
NUM_STABILIZERS = DISTANCE**2 - 1  # 8 for d=3

# --- 1. DATA ADAPTER ---
def get_batch(batch_size=BATCH_SIZE):
    """
    Fetches data from your generator and formats it for JAX.
    Returns:
        syndromes: (Batch, Rounds, NUM_STABILIZERS, 4)
        targets: (Batch, 1)
    """
    # 1. Generate Raw Data
    # Returns: post_1, events_1, post_2, events_2, logical_errors
    p1, e1, p2, e2, labels = generate_training_data(
        d=DISTANCE, 
        rounds=ROUNDS, 
        shots=batch_size, 
        p=0.005 # Physical error rate
    )
    
    # 2. Reshape Logic
    # The generator returns flattened arrays (Batch, Total_Measurements).
    # We need (Batch, Rounds, Num_Stabilizers).
    limit = ROUNDS * NUM_STABILIZERS
    
    # post_1/post_2 include final data qubit readout — slice to bulk rounds only
    # Shape becomes: (Batch, Rounds, NUM_STABILIZERS)
    p1_in = p1[:, :limit].reshape(batch_size, ROUNDS, NUM_STABILIZERS)
    p2_in = p2[:, :limit].reshape(batch_size, ROUNDS, NUM_STABILIZERS)
    
    # events_1/events_2 are already exactly (Batch, Rounds * NUM_STABILIZERS) — reshape directly
    e1_in = e1.reshape(batch_size, ROUNDS, NUM_STABILIZERS)
    e2_in = e2.reshape(batch_size, ROUNDS, NUM_STABILIZERS)
    
    # 3. Stack Channels
    # We stack them along the last dimension to create a feature vector.
    # Final Shape: (Batch, Rounds, NUM_STABILIZERS, 4)
    syndromes = np.stack([p1_in, e1_in, p2_in, e2_in], axis=-1)
    
    # 4. Format Labels
    # Labels need to be (Batch, 1) float32
    targets = labels.astype(np.float32).reshape(batch_size, 1)
    
    return jnp.array(syndromes), jnp.array(targets)

# --- 2. THE MODEL WRAPPER ---
def unroll_model(syndromes):
    """
    syndromes: (Batch, Rounds, NUM_STABILIZERS, 4)
    """
    # Initialize the architecture
    cycle_model = CycleArchitecture(mixing_mult=0.5, output_size=64, distance=DISTANCE)
    
    # Initial Decoder State: (Batch, NUM_STABILIZERS, 64)
    batch_size = syndromes.shape[0]
    d_state = jnp.zeros((batch_size, NUM_STABILIZERS, 64))
    
    final_logit = None
    
    # Loop over time
    for t in range(syndromes.shape[1]):
        # Current input: (Batch, NUM_STABILIZERS, 4)
        current_check = syndromes[:, t, :, :]
        
        # Run RNN step
        d_state, logit = cycle_model(current_check, d_state)
        final_logit = logit

    return final_logit

# Create the pure functions
network = hk.transform(unroll_model)

# Optimizer with gradient clipping for transformer training stability
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(LEARNING_RATE),
)

# --- 3. LOSS & UPDATE FUNCTIONS ---
def loss_fn(params, rng, syndromes, targets):
    logits = network.apply(params, rng, syndromes)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets))

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
    predictions = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    accuracy = jnp.mean(predictions == targets)
    return loss, accuracy

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
sample_input, sample_target = get_batch() # Get real shapes from real data

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

# Training
print("Starting training...")
best_eval_loss = float("inf")

try:
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # 1. Generate Fresh Data (Online Training)
        # We generate new random circuits every step to prevent overfitting
        syndromes, targets = get_batch()
        
        rng, step_rng = jax.random.split(rng)
        
        # 2. Update Weights
        params, opt_state, loss_val = update_step(
            params, 
            opt_state, 
            step_rng, 
            syndromes, 
            targets
        )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss_val:.4f}")

        # 3. Evaluation on held-out data
        if (epoch + 1) % EVAL_EVERY == 0:
            rng, eval_rng = jax.random.split(rng)
            eval_syndromes, eval_targets = get_batch(batch_size=EVAL_SHOTS)
            eval_loss, eval_acc = eval_step(params, eval_rng, eval_syndromes, eval_targets)
            print(f"  [EVAL] Epoch {epoch}: Loss = {eval_loss:.4f}, Accuracy = {eval_acc:.4f}")
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(params, opt_state, epoch, os.path.join(CHECKPOINT_DIR, "best.pkl"))

        # 4. Periodic checkpointing
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, opt_state, epoch, latest_ckpt)

except KeyboardInterrupt:
    print("\nTraining stopped manually.")
    save_checkpoint(params, opt_state, epoch, latest_ckpt)

print("Done.")
