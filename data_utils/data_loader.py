import numpy as np
import jax.numpy as jnp

# Imports from your sibling directory 'data_generation_types'
# These assume you are running the code from the root 'AlphaQubit_Project/'
from data_generation_types.generate_SI1000_data import generate_SI1000_training_data
from data_generation_types.generate_DEM_data import generate_pij_DEM_data 

class CurriculumDataLoader:
    def __init__(self, config,):
        """
        Args:
            config: Your configuration object.
            mode: "SI1000" (Generative Pretraining) or "DEM" (Device Specific). This is in config.
        """
        self.config = config
        self.mode = config.mode
        self.d = config.code_distance
        self.rounds = config.rounds
        # Calculate number of stabilizers (d^2 - 1)
        self.num_stabilizers = self.d**2 - 1
        
        # [cite_start]--- Curriculum Settings [cite: 81] ---
        # DEM Curriculum: Scale factor f from 0.5 to 1.0 (Training details -> Sycamore data)
        self.dem_min_f = 0.5
        self.dem_max_f = 1.0
        
        # SI1000 Curriculum: Physical error p from 0.001 to 0.005 (Example range)
        # Note: Adjust these p values based on what your generate_SI1000_data expects
        self.si_min_p = 0.001
        self.si_max_p = 0.005

    def _get_difficulty_params(self, progress):
        """
        Calculates current noise parameters based on training progress (0.0 to 1.0).
        """
        # Linear ramp is a simple, effective approximation of the curriculum 
        current_f = self.dem_min_f + (self.dem_max_f - self.dem_min_f) * progress
        current_p = self.si_min_p + (self.si_max_p - self.si_min_p) * progress
        return current_f, current_p

    def get_batch(self, batch_size, epoch, total_epochs):
        """
        Generates a batch of standardized data.
        Returns: 
            syndromes: (Batch, Rounds, Num_Stabilizers, 4)
            targets: (Batch, Rounds, 1)
        """
        # 1. Calculate Curriculum Progress
        # Clamp progress between 0 and 1
        if total_epochs > 0:
            progress = min(max(epoch / total_epochs, 0.0), 1.0)
        else:
            progress = 1.0
            
        curr_f, curr_p = self._get_difficulty_params(progress)

        # 2. Generate Raw Data based on Mode
        if self.mode == "SI1000":
            # SI1000 returns 4 channels: Meas, Event, LeakMeas, LeakEvent
            p1, e1, p2, e2, labels = generate_SI1000_training_data(
                d=self.d, rounds=self.rounds, shots=batch_size, p=curr_p
            )
            
        elif self.mode == "DEM":
            # DEM returns 2 channels: Meas, Event (Leakage is 0)
            # Ensure generate_pij_DEM_data accepts 'scale_factor' as discussed
            p1, e1, labels = generate_pij_DEM_data(
                d=self.d, rounds=self.rounds, shots=batch_size, scale_factor=curr_f
            )
            
            # Create dummy leakage channels (Zeros) for DEM data
            # (Leakage is not simulated in standard Stim DEMs, but valid inputs are needed)
            p2 = np.zeros_like(p1)
            e2 = np.zeros_like(e1)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 3. Reshape and Format (Standardize Shape)
        # We process the flattened output into (Batch, Rounds, Stabilizers)
        limit = self.rounds * self.num_stabilizers
        
        # Slice to ensure we only have the bulk rounds if generation creates extras
        p1_in = p1[:, :limit].reshape(batch_size, self.rounds, self.num_stabilizers)
        e1_in = e1[:, :limit].reshape(batch_size, self.rounds, self.num_stabilizers)
        p2_in = p2[:, :limit].reshape(batch_size, self.rounds, self.num_stabilizers)
        e2_in = e2[:, :limit].reshape(batch_size, self.rounds, self.num_stabilizers)

        # 4. Stack Channels
        # Order: (Events_1, Meas_1, Events_2, Meas_2) — consistent across SI1000, DEM, and hardware
        # Result: (Batch, Rounds, Stabilizers, 4)
        syndromes = np.stack([e1_in, p1_in, e2_in, p2_in], axis=-1)
        
        # Ensure labels are (Batch, Rounds, 1) and float32
        # SI1000: labels already (batch_size, rounds). DEM: (batch_size, 1) -> broadcast.
        if labels.ndim == 1 or labels.shape[1] == 1:
            labels = np.broadcast_to(
                np.asarray(labels).reshape(batch_size, -1)[:, 0:1],
                (batch_size, self.rounds)
            ).copy()
        targets = labels.astype(np.float32).reshape(batch_size, self.rounds, 1)

        return jnp.array(syndromes), jnp.array(targets)