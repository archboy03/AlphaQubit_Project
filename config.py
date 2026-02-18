# FILE FOR CENTRALIZING HYPERPARAMETERS
# Hardware profile: RTX 3050 Ti (4 GB VRAM), 7.6 GB RAM, i7-11800H
# Estimated time: ~45 min pre-train + ~30 min fine-tune + ~5 min test

from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Surface code parameters
    code_distance: int = 3
    code_task: str = "surface_code:rotated_memory_z"
    rounds: int = 25      # Match test round so pre-training sees full-length sequences
    shots: int = 100       # Default for standalone generator calls only;
                            # training/finetuning use batch_size instead

    # Model
    hidden_size: int = 64       # Dimension of stabilizer embeddings and transformer model
    num_layers: int = 3         # Number of stacked SyndromeTransformer layers in the RNN core
    mixing_mult: float = 0.7    # Fixed state/input scaling used in RNN core
    use_moe: bool = False       # Default to dense FFN; set True to enable MoE
    ffn_hidden_dim: int = 128   # Hidden width for dense FFN block
    moe_num_experts: int = 4    # Used only when use_moe=True
    moe_expert_dim: int = 128   # Used only when use_moe=True
    
    # Training (pre-training on SI1000)
    # batch=64 @ r=25 uses ~3.3 GB VRAM on RTX 3050 Ti
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_steps: int = 5000
    mode: str = "SI1000"        # Should be "SI1000" or "DEM"
    # SI1000 noise range: physical error rate p sampled uniformly per batch.
    # Stay below the ~1% threshold for d=3. Range [0.001, 0.005] gives meaningful
    # error rates without overwhelming the code's correction capacity.
    si_min_p: float = 0.001
    si_max_p: float = 0.005

    # Fine-tuning (hardware data)
    # batch=32 fits r=23 in ~3.3 GB VRAM; r=25 excluded -- reserved for testing only
    finetune_rounds: List[int] = field(
        default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    )
    test_round: int = 25
    finetune_lr: float = 1e-4
    finetune_weight_decay: float = 1e-4  # L2 regularization via AdamW
    finetune_steps: int = 2000
    finetune_batch_size: int = 32
    finetune_eval_every: int = 200
    finetune_log_every: int = 10
    finetune_checkpoint_every: int = 500
    early_stopping_patience: int = 5     # Stop after N evals with no improvement
    pretrained_checkpoint: str = "checkpoints/latest.pkl"  # Prefer latest pretrain run when iterating
    val_split: float = 0.1
    # Soften binary hardware inputs to better match SI1000's continuous [0,1] distribution.
    # 0.0 = no softening (raw 0/1). 0.05 = map 0->0.05, 1->0.95 for smoother transfer.
    finetune_input_softening: float = 0.05

    # Logging
    log_every: int = 10
    checkpoint_every: int = 500
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    data_dir: str = "data/"  # override this locally
