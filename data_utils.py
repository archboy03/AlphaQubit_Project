import stim 
import numpy as np

def generate_training_data(d=3, rounds=3, shots=100, noise_level=0.01):
    """Generate training data for a d-qubit stabilizer code."""

    # Create surface code blueprint

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        code_distance=2,
        after_clifford_depolarization=noise_level,
    )

    # Sample syndromes and logical errors
    sampler = circuit.compile_detector_sampler()
    syndromes, logical_errors = sampler.sample(shots=shots, seperate_observables=True)

    # syndromes: (Shots, Total_Detectors) -> 0 or 1
    # logical_errors: (Shots, 1) -> did the logical error occur?

    return syndromes.astype(np.float32), logical_errors.astype(np.float32)

    
    