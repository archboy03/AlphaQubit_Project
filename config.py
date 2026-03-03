# FILE FOR CENTRALIZING HYPERPARAMETERS

from dataclasses import dataclass

@dataclass
class Config:
    # Surface code parameters
    code_distance: int = 3
    code_task: str = "surface_code:rotated_memory_z"
    rounds: int = 25  # Use 25 for full training; 3 for faster testing
    shots: int = 100000

    # Model
    hidden_size: int = 64       # Dimension of stabilizer embeddings and transformer model
    num_layers: int = 3         # Number of stacked SyndromeTransformer layers in the RNN core
    
    # Training
    learning_rate: float = 5e-4
    batch_size: int = 32
    num_steps: int = 100000         # Set low for testing; use 100000 for full training
    mode: str = "SI1000"        # Should be "SI1000" or "DEM"
    positive_class_weight: float = 3.0   # Up-weight error class to combat class imbalance
    final_round_loss_weight: float = 3.0 # Extra weight on final-round prediction (the actual decoding target)

    # Curriculum noise range (SI1000 mode)
    si_min_p: float = 0.003     # Starting physical error rate (raised from 0.001)
    si_max_p: float = 0.008     # Ending physical error rate (raised from 0.005)
    
    # Logging
    log_every: int = 10
    checkpoint_every: int = 1000
    tensorboard_latest_only: bool = True   # If True, overwrite runs/latest each run; if False, keep timestamped runs
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    data_dir: str = "data/"  # override this locally

    # Finetuning
    finetune_learning_rate: float = 1e-4
    finetune_epochs: int = 1500            # Training steps per round folder
    finetune_checkpoint_dir: str = "checkpoints/finetune"
    finetune_tensorboard_dir: str = "runs/finetune"
    pretrained_checkpoint: str = "checkpoints/latest.pkl"
