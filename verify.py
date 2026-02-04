import jax
import haiku as hk

print("--- AlphaQubit Environment Check ---")
print(f"JAX Version: {jax.__version__}")
print(f"Haiku Version: {hk.__version__}")
print(f"GPU Detected: {jax.devices()}")