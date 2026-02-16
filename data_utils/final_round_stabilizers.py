# This is a file that calculates the final round stabilizers from the data qubit measurements
# ISSUES:
# The main issue here is understanding the output format from stim
# For the final round measurement stim returns the data qubits measurement, n*n of them
# We then want to calculate the stabilizer measurements from this, which is possible
# However, we don't know which data qubits are which elements of the 9D array and then also how the stabilizer checks correspond to these measurments

import numpy as np
from config import Config 

config = Config()

# This function only works for this specific type of surface code, rotated_memory_z, and currently only d = 3
# Maybe in the future could think about trying to create a function that can do this for any general circuit

def calculate_final_round_stabilizers(data_qubits, d = config.code_distance):
    # I have just done this manually which isn't very good, so need to think how to generalize this

    # Manually create code structure
    # Dictionary for Z-measurements, note we leave X measurements as a default -1 (chosen embedding for null measurement)
    # Keys = stabilizer number (1-indexed), Values = data qubit indices (0-indexed) that participate

    code_z_structure = {1: [0, 1], 3: [1, 2, 4, 5], 6: [3, 4, 6, 7], 8: [7, 8]}
    stabilizers = d*d - 1   # Number of stabilizers

    num_shots = data_qubits.shape[0]
    stabilizer_values = np.full((num_shots, stabilizers), -1, dtype=data_qubits.dtype)   # Default -1 for X measurements

    for z, idxs in code_z_structure.items():
        stabilizer_values[:, z-1] = np.sum(data_qubits[:, idxs], axis=1) % 2   # Parity of data qubit measurements
    
    return stabilizer_values
