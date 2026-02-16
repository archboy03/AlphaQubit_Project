import stim
import numpy as np
import os
from data_utils.scale_dem_probs import scale_dem_probabilities
from data_utils.detection_events_to_measurements import detection_events_to_measurements

try:
    from config_local import LocalConfig as Config
except ImportError:
    from config import Config

config = Config()


def generate_pij_DEM_data(
    scale_factor: float = 1.0,
    data_dir: str = config.data_dir,
    rounds: int = config.rounds,
    shots: int = config.shots,
    d: int = config.code_distance,
    fold: str = "even",  # "even" uses pij_from_even_for_odd.dem
                         # "odd"  uses pij_from_odd_for_even.dem
):
    """
    Generate binarized training data by sampling from a pre-fitted pij DEM.

    The Zenodo dataset contains one folder per round length, e.g.:
        surface_code_bZ_d3_r01_center_3_5/
        surface_code_bZ_d3_r03_center_3_5/
        ...
        surface_code_bZ_d3_r25_center_3_5/

    Each folder contains:
        pij_from_even_for_odd.dem  -- DEM fitted on even shots, decode odd
        pij_from_odd_for_even.dem  -- DEM fitted on odd shots, decode even

    Args:
        scale_factor: Factor to scale DEM error probabilities by (1.0 = no scaling)
        data_dir:   Path to the folder containing all round subfolders.
                    e.g. "/mnt/c/One Drive Files/Quantum Computing Project/z_centre_3_5_d3_data"
        rounds:     Number of error correction rounds (must be odd: 1,3,5,...,25)
        shots:      Number of shots to sample
        d:          Code distance
        fold:       Which pij DEM to load - "even" or "odd"

    Returns:
        detection_events:  np.array of shape (shots, num_detectors) dtype float32
                           Binary detection events, 1 = detector fired
        measurements:      np.array of shape (shots, (rounds + 1) * num_stabilizers) dtype float32
                           Per-round stabilizer measurements recovered from detection events,
                           including the initial all-zeros round. e.g. 32D for d=3, r=3.
        logical_errors:    np.array of shape (shots, 1) dtype float32
                           Binary logical observable flips, 1 = logical error
    """

    # --- Validate inputs ---
    assert fold in ("even", "odd"), "fold must be 'even' or 'odd'"
    assert rounds % 2 == 1 and 1 <= rounds <= 25, \
        "rounds must be odd and between 1 and 25"

    # --- Build path to the correct DEM file ---
    # Folder name format: surface_code_bZ_d3_r01_center_3_5
    round_str = f"r{rounds:02d}"
    folder_name = f"surface_code_bZ_d{d}_{round_str}_center_3_5"
    dem_filename = f"pij_from_{fold}_for_{'odd' if fold == 'even' else 'even'}.dem"
    dem_path = os.path.join(data_dir, folder_name, dem_filename)

    # --- Safety check ---
    assert os.path.exists(dem_path), \
        f"DEM file not found at: {dem_path}\n" \
        f"Check your data_dir and that rounds={rounds} is valid."

    # --- Load and sample from the DEM ---
    dem = stim.DetectorErrorModel.from_file(dem_path)

    # Apply scale factor here
    if scale_factor != 1.0:
        dem = scale_dem_probabilities(dem, scale_factor)

    sampler = dem.compile_sampler()

    # sample() returns a 3-tuple: (detector_data, obs_data, error_data)
    # error_data is None when return_errors=False (the default)
    detection_events, observable_flips, _ = sampler.sample(shots=shots)

    # --- Convert to float32 ---
    detection_events = detection_events.astype(np.float32)
    logical_errors = observable_flips.astype(np.float32)

    # --- Recover per-round measurements from detection events ---
    num_stabilizers = d ** 2 - 1
    measurements = detection_events_to_measurements(detection_events, rounds, num_stabilizers)

    return detection_events, measurements, logical_errors