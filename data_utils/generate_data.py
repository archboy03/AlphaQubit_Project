import stim
import numpy as np
from .noise_model import SoftNoiseModel
from config import Config 

config = Config()

# Note: All documentation for stim can be found on github in doc -> python_api_reference_vDev.md 

def generate_training_data(p, d=config.code_distance, rounds=config.rounds, shots=config.shots, code_task = config.code_task):
    """Generate training data with soft information."""
    
    # 1. Create Stim circuit
    # Note: We set measurement error to 0 here because we will simulate 
    # the realistic measurement noise manually using the SoftNoiseModel.

    # stim.Circuit.generated generates common circuits such as the rotated_memory_z code we are using
    circuit = stim.Circuit.generated(
        code_task,
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,          
        after_reset_flip_probability=2*p,        
        before_measure_flip_probability=0.0,      # Set to 0.0 (handled by Soft Model)
        before_round_data_depolarization=p/10,    # Idle noise on data qubits between rounds
    )

    # We generate a random seed explicitly so we can pass it to both samplers.
    # This ensures the "Raw Measurements" and "Logical Errors" come from the 
    # exact same simulation trajectory.

    # Note: We need to do this as compile_detector_sampler.sample() returns syndromes (parity checks) not the raw data we want for our model
    current_seed = np.random.randint(0, 2**31 - 1)

    # 2. Get Ground Truth / Raw Data
    # Pass the seed here
    sampler = circuit.compile_sampler(seed=current_seed)
    raw_measurements = sampler.sample(shots=shots).astype(np.float32)

    # Get logical labels
    # Pass the SAME seed here
    detector_sampler = circuit.compile_detector_sampler(seed=current_seed)
    _, logical_errors = detector_sampler.sample(shots=shots, separate_observables=True)
    
    # 3. Apply Soft Noise Model 
    noise_model = SoftNoiseModel(snr=10.0, t_integration=0.01)
    z_values = noise_model.generate_signal(raw_measurements)
    post_1, post_2 = noise_model.calculate_posteriors(z_values)
    
    # Calculate shapes dynamically to be safe
    # For rotated memory z, measurements = rounds * (d^2-1) + d^2
    num_stabilizers = d**2 - 1 
    
    end_of_bulk = rounds * num_stabilizers
    
    # Safety Check: Ensure the circuit actually produced the expected number of bits
    assert raw_measurements.shape[1] == end_of_bulk + (d*d), \
        f"Mismatch in expected circuit output. Expected {end_of_bulk + d*d}, got {raw_measurements.shape[1]}"

    bulk_post_1 = post_1[:, :end_of_bulk].reshape(shots, rounds, num_stabilizers)
    bulk_post_2 = post_2[:, :end_of_bulk].reshape(shots, rounds, num_stabilizers)

    prev_post_1 = np.zeros_like(bulk_post_1)
    prev_post_1[:, 1:, :] = bulk_post_1[:, :-1, :] 
    
    prev_post_2 = np.zeros_like(bulk_post_2)
    prev_post_2[:, 1:, :] = bulk_post_2[:, :-1, :] 

    events_1 = noise_model.soft_xor(bulk_post_1, prev_post_1)
    events_2 = noise_model.soft_xor(bulk_post_2, prev_post_2)

    events_1 = events_1.reshape(shots, -1)
    events_2 = events_2.reshape(shots, -1)

    num_final_qubits = d * d
    post_1[:, -num_final_qubits:] = (post_1[:, -num_final_qubits:] > 0.5).astype(np.float32)
    post_2[:, -num_final_qubits:] = (post_2[:, -num_final_qubits:] > 0.5).astype(np.float32)

    return post_1, events_1, post_2, events_2, logical_errors