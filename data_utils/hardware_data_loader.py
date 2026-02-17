import numpy as np
import jax.numpy as jnp
from data_generation_types.load_hardware_data import load_hardware_data


class HardwareDataLoader:
    """
    Batch sampler for fine-tuning on real experimental hardware data.

    Pre-loads all specified round folders into memory and provides
    random batch sampling with train/validation splitting.
    """

    def __init__(self, config, split="train"):
        """
        Args:
            config:  Config object with data_dir, code_distance, finetune_rounds,
                     and val_split fields.
            split:   "train" or "val" -- which portion of the data to serve.
        """
        self.config = config
        self.split = split
        self.d = config.code_distance
        self.num_stabilizers = self.d ** 2 - 1
        self.val_split = config.val_split
        self.round_counts = config.finetune_rounds

        # Pre-load all round folders
        self._data = {}
        print(f"Loading hardware data ({split} split)...")
        for r in self.round_counts:
            det_events, measurements, obs_flips = load_hardware_data(
                data_dir=config.data_dir,
                rounds=r,
                d=self.d,
            )
            num_shots = det_events.shape[0]
            split_idx = int(num_shots * (1.0 - self.val_split))

            if split == "train":
                s = slice(0, split_idx)
            else:
                s = slice(split_idx, num_shots)

            self._data[r] = {
                "detection_events": det_events[s],
                "measurements": measurements[s],
                "obs_flips": obs_flips[s],
            }
            n = det_events[s].shape[0]
            print(f"  r={r:02d}: {n} shots loaded")

        print(f"Hardware data loading complete. Rounds: {self.round_counts}")

    def num_shots(self, rounds):
        """Return the number of available shots for a given round count."""
        return self._data[rounds]["detection_events"].shape[0]

    def get_batch(self, batch_size, rounds):
        """
        Sample a random batch for a specific round count.

        Args:
            batch_size: Number of shots to sample.
            rounds:     Which round count to sample from.

        Returns:
            syndromes: jnp.array of shape (batch_size, rounds, num_stabilizers, 4)
            targets:   jnp.array of shape (batch_size, rounds, 1)
        """
        assert rounds in self._data, (
            f"Round count {rounds} not loaded. Available: {list(self._data.keys())}"
        )
        data = self._data[rounds]
        n = data["detection_events"].shape[0]
        indices = np.random.choice(n, size=batch_size, replace=(batch_size > n))

        det_events = data["detection_events"][indices]
        measurements = data["measurements"][indices]
        obs_flips = data["obs_flips"][indices]

        return self._format_batch(det_events, measurements, obs_flips, rounds, batch_size)

    def get_batch_mixed(self, batch_size):
        """
        Sample a batch from a randomly chosen round count.

        Returns:
            syndromes, targets (same shapes as get_batch), and the chosen round count.
        """
        rounds = int(np.random.choice(self.round_counts))
        syndromes, targets = self.get_batch(batch_size, rounds)
        return syndromes, targets, rounds

    def get_all(self, rounds):
        """
        Return ALL shots for a given round count (for full evaluation).

        Returns:
            syndromes: jnp.array of shape (num_shots, rounds, num_stabilizers, 4)
            targets:   jnp.array of shape (num_shots, rounds, 1)
        """
        assert rounds in self._data, (
            f"Round count {rounds} not loaded. Available: {list(self._data.keys())}"
        )
        data = self._data[rounds]
        n = data["detection_events"].shape[0]
        return self._format_batch(
            data["detection_events"],
            data["measurements"],
            data["obs_flips"],
            rounds,
            n,
        )

    def _format_batch(self, det_events, measurements, obs_flips, rounds, batch_size):
        """
        Convert raw arrays into the model's standard input format.

        Channels: (detection_events, measurements, zeros, zeros)
        matching the DEM mode convention in CurriculumDataLoader.

        Args:
            det_events:   (batch_size, num_detectors) float32
            measurements: (batch_size, (rounds+1)*num_stabilizers) float32
            obs_flips:    (batch_size,) float32
            rounds:       number of QEC rounds
            batch_size:   number of shots

        Returns:
            syndromes: jnp.array (batch_size, rounds, num_stabilizers, 4)
            targets:   jnp.array (batch_size, rounds, 1)
        """
        limit = rounds * self.num_stabilizers

        # Channel 0: detection events reshaped to (B, R, S)
        p1 = det_events[:, :limit].reshape(batch_size, rounds, self.num_stabilizers)

        # Channel 1: measurements (skip initial zeros round, take R rounds)
        # measurements has (rounds+1)*num_stabilizers values: M[0], M[1], ..., M[rounds]
        # Skip M[0] (the initial zeros) to get M[1] through M[rounds]
        meas_reshaped = measurements.reshape(batch_size, rounds + 1, self.num_stabilizers)
        e1 = meas_reshaped[:, 1:, :]  # (B, R, S) -- skip the initial zero round

        # Channels 2-3: zeros (no leakage info in experimental data)
        p2 = np.zeros_like(p1)
        e2 = np.zeros_like(p1)

        # Stack into (B, R, S, 4)
        syndromes = np.stack([p1, e1, p2, e2], axis=-1)

        # Broadcast single observable label to all rounds: (B, R, 1)
        labels = np.broadcast_to(
            obs_flips.reshape(batch_size, 1, 1),
            (batch_size, rounds, 1),
        ).copy().astype(np.float32)

        return jnp.array(syndromes), jnp.array(labels)
