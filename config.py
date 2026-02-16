# FILE FOR CENTRALIZING HYPERPARAMETERS

from dataclasses import dataclass

@dataclass
class Config:
    # Surface code parameters
    code_distance: int = 3
    code_task: str = "surface_code:rotated_memory_z"
    rounds: int = 3  # Use 25 for full training; 3 for faster testing
    shots: int = 100

    # Model
    hidden_size: int = 64       # Dimension of stabilizer embeddings and transformer model
    num_layers: int = 3         # Number of stacked SyndromeTransformer layers in the RNN core
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_steps: int = 50         # Set low for testing; use 100000 for full training
    mode: str = "SI1000"        # Should be "SI1000" or "DEM"
    
    # Logging
    log_every: int = 10
    checkpoint_every: int = 1000
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    data_dir: str = "data/"  # override this locally
