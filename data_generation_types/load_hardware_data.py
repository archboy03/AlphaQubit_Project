import stim
import numpy as np
import os
import yaml
from data_utils.detection_events_to_measurements import detection_events_to_measurements


def load_hardware_data(
    data_dir: str,
    rounds: int,
    d: int = 3,
    center_row: int = 3,
    center_col: int = 5,
    basis: str = "Z",
):
    """
    Load actual experimental data from a Zenodo-format round folder.

    Reads detection_events.b8 and obs_flips_actual.01 from the appropriate
    subdirectory and converts them into numpy arrays matching the format
    returned by generate_pij_DEM_data.

    Args:
        data_dir:   Root directory containing all round subfolders.
        rounds:     Number of QEC rounds (must be odd: 1,3,...,25).
        d:          Code distance.
        center_row: Row coordinate of center data qubit.
        center_col: Column coordinate of center data qubit.
        basis:      Measurement basis ("Z" or "X").

    Returns:
        detection_events: np.array (num_shots, num_detectors), float32.
        measurements:     np.array (num_shots, (rounds+1)*num_stabilizers), float32.
        obs_flips:        np.array (num_shots,), float32.  Binary 0/1.
    """
    folder_name = f"surface_code_b{basis}_d{d}_r{rounds:02d}_center_{center_row}_{center_col}"
    folder_path = os.path.join(data_dir, folder_name)

    assert os.path.isdir(folder_path), (
        f"Data folder not found: {folder_path}\n"
        f"Check data_dir and rounds={rounds}."
    )

    # --- Read metadata from properties.yml ---
    props_path = os.path.join(folder_path, "properties.yml")
    with open(props_path, "r") as f:
        props = yaml.safe_load(f)

    num_shots = props["shots"]
    num_detectors = props["circuit_detectors"]
    num_observables = props.get("circuit_observables", 1)

    # --- Load detection events (b8 format) ---
    det_path = os.path.join(folder_path, "detection_events.b8")
    detection_events = stim.read_shot_data_file(
        path=det_path,
        format="b8",
        num_detectors=num_detectors,
        num_observables=0,
    ).astype(np.float32)

    assert detection_events.shape == (num_shots, num_detectors), (
        f"Detection events shape mismatch: expected ({num_shots}, {num_detectors}), "
        f"got {detection_events.shape}"
    )

    # --- Load observable flips (01 text format) ---
    obs_path = os.path.join(folder_path, "obs_flips_actual.01")
    obs_flips = _read_01_file(obs_path, num_observables)

    assert obs_flips.shape[0] == num_shots, (
        f"Observable flips count mismatch: expected {num_shots}, "
        f"got {obs_flips.shape[0]}"
    )

    # --- Convert detection events to per-round measurements ---
    num_stabilizers = d ** 2 - 1
    measurements = detection_events_to_measurements(
        detection_events, rounds, num_stabilizers
    )

    return detection_events, measurements, obs_flips


def load_baseline_predictions(
    data_dir: str,
    rounds: int,
    d: int = 3,
    center_row: int = 3,
    center_col: int = 5,
    basis: str = "Z",
):
    """
    Load predictions from all baseline decoders in a round folder.

    Returns:
        dict mapping decoder name -> np.array (num_shots,) of float32 predictions.
    """
    folder_name = f"surface_code_b{basis}_d{d}_r{rounds:02d}_center_{center_row}_{center_col}"
    folder_path = os.path.join(data_dir, folder_name)

    decoders = [
        "pymatching",
        "correlated_matching",
        "belief_matching",
        "tensor_network_contraction",
    ]

    predictions = {}
    for decoder in decoders:
        filename = f"obs_flips_predicted_by_{decoder}.01"
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            predictions[decoder] = _read_01_file(filepath, num_observables=1)

    return predictions


def _read_01_file(path: str, num_observables: int = 1) -> np.ndarray:
    """
    Read a stim .01 format file.

    Each line contains one character per observable ('0' or '1').
    Returns a float32 array of shape (num_shots,) when num_observables=1,
    or (num_shots, num_observables) otherwise.
    """
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")

    if num_observables == 1:
        data = np.array([int(line.strip()) for line in lines], dtype=np.float32)
    else:
        data = np.array(
            [[int(c) for c in line.strip()] for line in lines], dtype=np.float32
        )

    return data
