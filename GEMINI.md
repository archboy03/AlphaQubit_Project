# AlphaQubit Project Context

This project is an implementation of a transformer-based Recurrent Neural Network (RNN) decoder for quantum error correction, inspired by the Google DeepMind and Google Quantum AI paper "[Real-time quantum error correction beyond break-even with a multi-scale decoder](https://www.nature.com/articles/s41586-024-08148-8)".

## Project Overview
The core goal is to decode error syndromes for surface codes (specifically $d=3$ rotated Z-memory) on the Sycamore chip. It leverages machine learning to handle realistic, hardware-dependent noise that traditional algorithms like Minimum Weight Perfect Matching (MWPM) might not fully capture.

### Key Technologies
- **Framework:** JAX with Haiku (`dm-haiku`) for neural network layers and Optax for optimization.
- **Quantum Simulation:** `stim` for generating synthetic error data.
- **Logging:** TensorBoard and `tensorboardX` for training visualization.

### Architecture
- **`CycleArchitecture`**: An RNN-based model that processes syndromes round-by-round.
- **`RNN_core`**: The central state-tracking component (SyndromeTransformer).
- **`StabilizerEmbedder`**: Maps syndrome measurements to the model's hidden dimension.
- **`ReadoutHead`**: Produces final error probability predictions.

---

## Building and Running

### Prerequisites
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Configuration
Hyperparameters and paths are centralized in `config.py`. To override settings locally (e.g., data directories), create a `config_local.py` file:
```python
from config import Config
class LocalConfig(Config):
    data_dir = "/your/local/path/to/data"
```

### Training
Start the training loop (which includes online data generation and curriculum learning):
```bash
python train.py
```
- **Curriculum Learning:** The `CurriculumDataLoader` automatically increases the noise level (`p` or `scale_factor`) as training progresses.
- **Monitoring:** TensorBoard starts automatically on port `6006`.

### Verification
Run a quick environment check:
```bash
python verify.py
```

---

## Development Conventions

### Coding Style
- **JAX-Native:** Use `jax.numpy` (jnp) for model logic and `numpy` (np) for data generation/loading.
- **Haiku Modules:** All model components should inherit from `hk.Module` and be transformed using `hk.transform` or `hk.multi_transform`.
- **Statelessness:** Adhere to JAX's functional programming paradigm (pass `rng` keys and `params` explicitly).

### Directory Structure
- `model/`: Architecture definitions (`model.py`, `model_components.py`).
- `data_utils/`: Data loading, scaling, and processing logic.
- `data_generation_types/`: Scripts for SI1000 (synthetic) and DEM (device-specific) data generation.
- `checkpoints/`: Model weights saved as `.pkl` files.
- `runs/`: TensorBoard event files.
- `utils/`: Metrics and visualization tools.

### Checkpointing
- Checkpoints are saved periodically (defined by `config.checkpoint_every`).
- The training script automatically attempts to resume from `checkpoints/latest.pkl`.
- The best model (by evaluation loss) is saved as `checkpoints/best.pkl`.

### Data Pipeline
The project uses two primary data modes:
1. **SI1000:** Synthetic data generated using `stim` with a soft noise model.
2. **DEM (Detector Error Model):** Data derived from experimental Sycamore chip results.
