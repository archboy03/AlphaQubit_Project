import stim
import numpy as np
from data_utils.noise_model import SoftNoiseModel
from data_utils.final_round_stabilizers import calculate_final_round_stabilizers

try:
    from config_local import LocalConfig as Config
except ImportError:
    from config import Config

config = Config()

# Note: All documentation for stim can be found on github in doc -> python_api_reference_vDev.md 

def generate_SI1000_training_data(p: np.float32, d=config.code_distance, rounds=config.rounds, shots=config.shots, code_task = config.code_task):
    """Generate training data with soft information.
    
    Returns logical_errors with shape (shots, rounds). Each column t is the true
    logical observable at round t+1 (what it would be if the experiment had stopped
    and been measured at that round). Uses circuits with rounds=1,2,...,N and the
    same seed so trajectories are consistent.
    """
    
    # We generate a random seed explicitly so we can pass it to all samplers.
    # This ensures the "Raw Measurements" and "Logical Errors" come from the
    # exact same simulation trajectory.
    current_seed = np.random.randint(0, 2**31 - 1)

    # 1. Create full Stim circuit for raw measurements
    # Note: We set measurement error to 0 here because we will simulate
    # the realistic measurement noise manually using the SoftNoiseModel.
    circuit_full = stim.Circuit.generated(
        code_task,
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=2*p,
        before_measure_flip_probability=0.0,      # Set to 0.0 (handled by Soft Model)
        before_round_data_depolarization=p/10,   # Idle noise on data qubits between rounds
    )

    # 2. Get Raw Data from full circuit
    sampler = circuit_full.compile_sampler(seed=current_seed)
    raw_measurements = sampler.sample(shots=shots).astype(np.float32)

    # 3. Get per-round logical labels: run circuits with rounds=1,2,...,N (same seed)
    # Each circuit yields the logical observable at that round from the same trajectory.
    logical_errors_list = []
    for r in range(1, rounds + 1):
        circuit_r = stim.Circuit.generated(
            code_task,
            distance=d,
            rounds=r,
            after_clifford_depolarization=p,
            after_reset_flip_probability=2*p,
            before_measure_flip_probability=0.0,
            before_round_data_depolarization=p/10,
        )
        detector_sampler = circuit_r.compile_detector_sampler(seed=current_seed)
        _, obs_r = detector_sampler.sample(shots=shots, separate_observables=True)
        logical_errors_list.append(np.asarray(obs_r).reshape(shots, -1)[:, 0])
    logical_errors = np.stack(logical_errors_list, axis=1).astype(np.float32)

    # 4. Apply Soft Noise Model 
    noise_model = SoftNoiseModel(snr=10.0, t_integration=0.01)
    z_values = noise_model.generate_signal(raw_measurements)
    post_1, post_2 = noise_model.calculate_posteriors(z_values)
    
    # Calculate shapes dynamically to be safe
    # For rotated memory z, measurements = rounds * (d^2-1) + d^2
    num_stabilizers = d**2 - 1 
    
    end_of_bulk = rounds * num_stabilizers

    # Expected total number of measurements in raw_measurements (per shot)
    expected_total = end_of_bulk + (d*d)
    
    # Safety Check: Ensure the circuit actually produced the expected number of bits
    assert raw_measurements.shape[1] == expected_total, \
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

    # Threshold so can't learn inverse of SoftXOR
    num_final_qubits = d * d
    post_1[:, -num_final_qubits:] = (post_1[:, -num_final_qubits:] > 0.5).astype(np.float32)
    post_2[:, -num_final_qubits:] = (post_2[:, -num_final_qubits:] > 0.5).astype(np.float32)

    # Now convert the data qubits into stabilizer measurements
    final_stabilizers_1 = calculate_final_round_stabilizers(post_1[:, -num_final_qubits:], d)
    final_stabilizers_2 = np.zeros((shots, num_final_qubits-1))

    # Replace the last num_final_qubits columns with the num_final_qubits - 1 stabilizer values
    post_1 = np.concatenate([post_1[:, :-num_final_qubits], final_stabilizers_1], axis=1).astype(np.float32)
    post_2 = np.concatenate([post_2[:, :-num_final_qubits], final_stabilizers_2], axis=1).astype(np.float32)

    return post_1, events_1, post_2, events_2, logical_errors

# Note on return format:
# post_1 is a 32D array with last entries binarized
# post_2 is a 32D array with last entries binarized (set to zero as leakage probs are low)
# events_1 is a 24D array of floats
# events_2 is a 24D array of floats