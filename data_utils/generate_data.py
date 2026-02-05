import stim
import numpy as np
from noise_model import SoftNoiseModel


def generate_training_data(d=3, rounds=3, shots=100, p=0.01):
    """Generate training data with soft information."""
    
    # 1. Create Stim circuit
    # Note: We set measurement error to 0 here because we will simulate 
    # the realistic measurement noise manually using the SoftNoiseModel.
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,          
        after_reset_flip_probability=2*p,        
        before_measure_flip_probability=0.0,      # Set to 0.0 (handled by Soft Model)
        after_horizontal_depolarization=p/10,     
        after_vertical_depolarization=p/10        
    )

    # 2. Get Ground Truth / Raw Data
    sampler = circuit.compile_sampler()
    # These are the "true" qubit states arriving at the readout line
    raw_measurements = sampler.sample(shots=shots).astype(np.float32)
    
    # Get logical labels (ground truth for training)
    detector_sampler = circuit.compile_detector_sampler()
    _, logical_errors = detector_sampler.sample(shots=shots, separate_observables=True)
    
    # 3. Apply Soft Noise Model
    noise_model = SoftNoiseModel(snr=10.0, t_integration=0.01)
    
    # A. Generate analog 'z' values (Simulate Physics)
    z_values = noise_model.generate_signal(raw_measurements)
    
    # B. Calculate Posteriors (What the decoder sees)
    # post_1: Probability measurement is 1 (conditioned on no leakage)
    # post_2: Probability of leakage
    post_1, post_2 = noise_model.calculate_posteriors(z_values)
    
    # C. Safety: Threshold the FINAL round [cite: 663]
    # The paper explicitly warns that giving soft info for the final round 
    # allows the network to cheat. We must hard-threshold the final round.
    # (Assuming the last N measurements correspond to the final round data qubits)
    num_final_qubits = d * d
    post_1[:, -num_final_qubits:] = (post_1[:, -num_final_qubits:] > 0.5).astype(np.float32)

    return post_1.astype(np.float32), post_2.astype(np.float32), logical_errors.astype(np.float32)
    