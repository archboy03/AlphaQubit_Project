# FILE FOR CENTRALIZING HYPERPARAMETERS

from dataclasses import dataclass

@dataclass
class Config:
    # Surface code parameters
    code_distance: int = 3
    code_task: str = "surface_code:rotated_memory_z"
    rounds: int = 3
    shots: int = 100

    # Model
    hidden_size: int = 256
    num_layers: int = 3
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_steps: int = 100000
    
    # LoggingWh
    log_every: int = 100
    checkpoint_every: int = 1000
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"