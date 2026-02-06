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
    z_values = noise_model.generate_signal(raw_measurements)
    post_1, post_2 = noise_model.calculate_posteriors(z_values)
    
    # We must reshape to [Shots, Rounds, Stabilizers_Per_Round] to difference across time.
    # Note: This reshaping logic depends on your specific circuit structure.
    # For a distance-3 memory experiment, you have 8 stabilizers per round.
    # The final round (data qubits) are NOT included in event calculation usually,
    # or are treated separately because they aren't stabilizer measurements.
    
    num_stabilizers = d**2 - 1  # 8 for d=3
    
    # Slice out just the bulk rounds (exclude final data qubit readout for event calc)
    # We assume 'post_1' is flat: [Round1_Stabs, Round2_Stabs, ..., Final_Data_Qubits]
    
    # 1. Separate Bulk Stabilizers from Final Data Readout
    # Total measurements = (Rounds * num_stabilizers) + (d * d)
    end_of_bulk = rounds * num_stabilizers
    
    bulk_post_1 = post_1[:, :end_of_bulk].reshape(shots, rounds, num_stabilizers)
    bulk_post_2 = post_2[:, :end_of_bulk].reshape(shots, rounds, num_stabilizers)

    # 2. Compute Events (Soft XOR across time axis=1)
    # First round events compare against "0" (or previous known state)
    # We pad with 0 (assuming perfectly healthy/0 state before experiment starts)
    prev_post_1 = np.zeros_like(bulk_post_1)
    prev_post_1[:, 1:, :] = bulk_post_1[:, :-1, :]  # Shift right
    
    prev_post_2 = np.zeros_like(bulk_post_2)
    prev_post_2[:, 1:, :] = bulk_post_2[:, :-1, :]  # Shift right

    # INPUT #2: Soft Detection Events
    events_1 = noise_model.soft_xor(bulk_post_1, prev_post_1)
    
    # INPUT #4: Event Leakage (Did leakage start or stop?)
    events_2 = noise_model.soft_xor(bulk_post_2, prev_post_2)

    # Flatten back to match your output structure if desired
    events_1 = events_1.reshape(shots, -1)
    events_2 = events_2.reshape(shots, -1)

    # --- Apply Final Round Thresholding (Anti-Cheating) ---
    num_final_qubits = d * d
    post_1[:, -num_final_qubits:] = (post_1[:, -num_final_qubits:] > 0.5).astype(np.float32)
    post_2[:, -num_final_qubits:] = (post_2[:, -num_final_qubits:] > 0.5).astype(np.float32)

    # Return all 4 inputs + label
    return post_1, events_1, post_2, events_2, logical_errors
    