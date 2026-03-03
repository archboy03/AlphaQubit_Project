"""
Experimental Data Loader for Sycamore chip data.

Loads pre-recorded detection events (.b8) and observable flips (.01)
from the experimental data folders, and formats them into the standard
4-channel syndrome format expected by the model.
"""

import os
import math
import numpy as np
import jax.numpy as jnp

from data_utils.detection_events_to_measurements import detection_events_to_measurements


class ExperimentalDataLoader:
    """Loads experimental data for a specific round count from disk.

    Each round folder (e.g. surface_code_bZ_d3_r03_center_3_5) contains:
      - detection_events.b8   : packed-bit binary detection events
      - obs_flips_actual.01   : ASCII '0'/'1' ground-truth observable flips

    The loader parses these files, recovers per-round measurements via
    cumulative XOR, and formats everything into the standard 4-channel
    syndrome tensor: (shots, rounds+1, num_stabilizers, 4).
    """

    def __init__(self, config, rounds: int, eval_fraction: float = 0.2, seed: int = 42):
        """
        Args:
            config:         Config object with at least `data_dir` and `code_distance`.
            rounds:         Number of QEC rounds to load (must be odd, 1-25).
            eval_fraction:  Fraction of shots reserved for evaluation.
            seed:           Random seed for the train/eval split.
        """
        self.d = config.code_distance
        self.rounds = rounds
        self.num_stabilizers = self.d ** 2 - 1

        # --- Build path ---
        folder_name = f"surface_code_bZ_d{self.d}_r{rounds:02d}_center_3_5"
        folder_path = os.path.join(config.data_dir, folder_name)
        assert os.path.exists(folder_path), (
            f"Experimental data folder not found: {folder_path}"
        )

        # --- Load detection events (.b8) ---
        det_path = os.path.join(folder_path, "detection_events.b8")
        detection_events = self._load_b8(det_path, self.rounds * self.num_stabilizers)

        # --- Load observable flips (.01) ---
        obs_path = os.path.join(folder_path, "obs_flips_actual.01")
        obs_flips = self._load_01(obs_path)

        total_shots = detection_events.shape[0]
        assert obs_flips.shape[0] == total_shots, (
            f"Shot count mismatch: detection_events has {total_shots}, "
            f"obs_flips has {obs_flips.shape[0]}"
        )

        # --- Recover measurements from detection events ---
        measurements = detection_events_to_measurements(
            detection_events, self.rounds, self.num_stabilizers
        )

        # --- Format into 4-channel syndromes ---
        # measurements: (shots, (rounds+1) * num_stabilizers) -> (shots, rounds+1, num_stabilizers)
        # detection events only cover rounds (not rounds+1), so we prepend a zero row
        syndromes = self._build_syndromes(detection_events, measurements)

        # --- Build targets: (shots, rounds+1, 1) ---
        # Single observable per shot, broadcast to all time steps
        targets = np.broadcast_to(
            obs_flips.reshape(-1, 1),
            (total_shots, self.rounds + 1),
        ).copy().astype(np.float32).reshape(total_shots, self.rounds + 1, 1)

        # --- Train / eval split ---
        rng = np.random.RandomState(seed)
        indices = rng.permutation(total_shots)
        split = int(total_shots * (1.0 - eval_fraction))

        train_idx = indices[:split]
        eval_idx = indices[split:]

        self.train_syndromes = syndromes[train_idx]
        self.train_targets = targets[train_idx]
        self.eval_syndromes = syndromes[eval_idx]
        self.eval_targets = targets[eval_idx]

        self._train_rng = np.random.RandomState(seed + 1)

        print(
            f"  ExperimentalDataLoader(r={rounds:02d}): "
            f"{len(train_idx)} train / {len(eval_idx)} eval shots, "
            f"syndrome shape per shot = {syndromes.shape[1:]}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batch(self, batch_size: int):
        """Sample a random training batch.

        Returns:
            syndromes: jnp.array of shape (batch_size, rounds+1, num_stabilizers, 4)
            targets:   jnp.array of shape (batch_size, rounds+1, 1)
        """
        n = self.train_syndromes.shape[0]
        idx = self._train_rng.choice(n, size=batch_size, replace=(batch_size > n))
        return jnp.array(self.train_syndromes[idx]), jnp.array(self.train_targets[idx])

    def get_eval_batch(self, batch_size: int):
        """Sample a random evaluation batch.

        Returns:
            syndromes: jnp.array of shape (batch_size, rounds+1, num_stabilizers, 4)
            targets:   jnp.array of shape (batch_size, rounds+1, 1)
        """
        n = self.eval_syndromes.shape[0]
        idx = self._train_rng.choice(n, size=batch_size, replace=(batch_size > n))
        return jnp.array(self.eval_syndromes[idx]), jnp.array(self.eval_targets[idx])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_b8(self, path: str, num_bits: int) -> np.ndarray:
        """Parse a Stim .b8 packed-bit file.

        Args:
            path:     Path to the .b8 file.
            num_bits: Number of meaningful bits per shot (= num_detectors).

        Returns:
            np.ndarray of shape (shots, num_bits), dtype float32, values 0.0/1.0.
        """
        bytes_per_shot = math.ceil(num_bits / 8)
        raw = np.fromfile(path, dtype=np.uint8)
        assert raw.size % bytes_per_shot == 0, (
            f"File size {raw.size} is not a multiple of {bytes_per_shot} bytes per shot"
        )
        raw = raw.reshape(-1, bytes_per_shot)
        unpacked = np.unpackbits(raw, axis=1, bitorder="little")[:, :num_bits]
        return unpacked.astype(np.float32)

    def _load_01(self, path: str) -> np.ndarray:
        """Parse a Stim .01 ASCII file (one '0' or '1' per line).

        Returns:
            np.ndarray of shape (shots,), dtype float32, values 0.0/1.0.
        """
        with open(path, "r") as f:
            data = f.read()
        values = np.array([int(ch) for ch in data if ch in ("0", "1")], dtype=np.float32)
        return values

    def _build_syndromes(
        self, detection_events: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """Build 4-channel syndrome tensor from detection events and measurements.

        Channels: [measurements, events, 0, 0]
        (Leakage channels are zeroed, matching the DEM path in CurriculumDataLoader.)

        Args:
            detection_events: (shots, rounds * num_stabilizers), float32
            measurements:     (shots, (rounds+1) * num_stabilizers), float32

        Returns:
            syndromes: (shots, rounds+1, num_stabilizers, 4), float32
        """
        shots = detection_events.shape[0]
        R1 = self.rounds + 1
        S = self.num_stabilizers

        # Measurements -> (shots, rounds+1, num_stabilizers)
        meas = measurements.reshape(shots, R1, S)

        # Detection events -> (shots, rounds, num_stabilizers)
        det = detection_events[:, : self.rounds * S].reshape(shots, self.rounds, S)
        # Prepend a zeros row for the initial round (no detection event for round 0)
        det_padded = np.concatenate(
            [np.zeros((shots, 1, S), dtype=np.float32), det], axis=1
        )  # (shots, rounds+1, num_stabilizers)

        # Leakage channels are zero (no leakage in experimental data)
        zeros = np.zeros((shots, R1, S), dtype=np.float32)

        # Stack: (shots, rounds+1, num_stabilizers, 4)
        syndromes = np.stack([meas, det_padded, zeros, zeros], axis=-1)
        return syndromes.astype(np.float32)


def discover_round_folders(config) -> list[int]:
    """Find all available round counts in the experimental data directory.

    Scans for folders matching the pattern surface_code_bZ_d{d}_r{RR}_center_3_5
    and returns a sorted list of round counts.

    Args:
        config: Config object with `data_dir` and `code_distance`.

    Returns:
        Sorted list of integer round counts, e.g. [1, 3, 5, ..., 25].
    """
    d = config.code_distance
    available_rounds = []
    if not os.path.exists(config.data_dir):
        print(f"Warning: data_dir does not exist: {config.data_dir}")
        return available_rounds

    for name in os.listdir(config.data_dir):
        prefix = f"surface_code_bZ_d{d}_r"
        suffix = f"_center_3_5"
        if name.startswith(prefix) and name.endswith(suffix):
            try:
                r_str = name[len(prefix) : name.index(suffix)]
                r = int(r_str)
                available_rounds.append(r)
            except (ValueError, IndexError):
                continue

    return sorted(available_rounds)
