import numpy as np
from scipy.special import erf

class SoftNoiseModel:
    def __init__(self, snr=10.0, t_integration=0.01, leakage_rate=0.00275):
        """
        Implements the AlphaQubit soft readout noise model.
        
        Args:
            snr (float): Signal-to-Noise Ratio. Default 10.0 matches the paper[cite: 840].
            t_integration (float): Normalized measurement time (t_meas / T1). 
                                   Controls decay tail (amplitude damping). Default 0.01[cite: 840].
            leakage_rate (float): Probability of injecting leakage |2> before measurement. 
                                  Default 0.275%[cite: 830].
        """
        self.snr = snr
        self.t = t_integration
        self.leakage_rate = leakage_rate
        
        # Precompute constants for PDF calculations
        self.sqrt_snr = np.sqrt(snr)
        self.inv_sqrt_pi = 1 / np.sqrt(np.pi)
        self.sigma = 1.0 / np.sqrt(2 * snr)  # Gaussian width derived from P0 formula
        
        # Priors for posterior calculation (approximated from leakage rate)
        self.w2 = leakage_rate
        self.w0 = (1 - self.w2) / 2
        self.w1 = (1 - self.w2) / 2

    def _pdf_0(self, z):
        """Probability Density Function for state |0> (Gaussian centered at 0)[cite: 600]."""
        # P0(z) = sqrt(SNR/pi) * exp(-SNR * z^2)
        return self.sqrt_snr * self.inv_sqrt_pi * np.exp(-self.snr * (z**2))

    def _pdf_1(self, z, scale_t=1.0):
        """
        Probability Density Function for state |1> with decay[cite: 601].
        Includes amplitude damping (decay to 0) and Gaussian noise.
        """
        t = self.t * scale_t
        
        # Term 1: Decay component (integration of exponential decay path)
        term1_pre = (t / 2) * np.exp(-t * (z - t / (4 * self.snr)))
        
        arg1 = self.sqrt_snr * (z - t / (2 * self.snr))
        arg2 = self.sqrt_snr * (1 - z + t / (2 * self.snr))
        term1_erf = erf(arg1) + erf(arg2)
        
        term1 = term1_pre * term1_erf
        
        # Term 2: No-decay component (Gaussian centered at 1, scaled by survival prob)
        term2 = np.exp(-t) * self.sqrt_snr * self.inv_sqrt_pi * np.exp(-self.snr * ((z - 1)**2))
        
        return term1 + term2

    def _pdf_2(self, z):
        """
        PDF for state |2> (Leakage).
        Modeled as P1 shifted by +1, with double the decay rate[cite: 602].
        """
        return self._pdf_1(z - 1, scale_t=2.0)

    def generate_signal(self, measurements):
        """
        Generates continuous 'z' values from binary measurements (Physics Simulation).
        """
        n_samples = measurements.size
        measurements_flat = measurements.flatten()
        
        # 1. Inject Leakage: Randomly overwrite 0s/1s with 2s [cite: 830]
        # We start with the binary outcome from Stim
        states = measurements_flat.copy().astype(int)
        
        # Determine which shots get leaked
        leak_mask = np.random.random(n_samples) < self.leakage_rate
        states[leak_mask] = 2
        
        # 2. Simulate Analog Signal Generation 
        # We simulate the physical decay process to generate 'z'
        # State |0>: Gaussian at 0
        z_values = np.random.normal(0, self.sigma, size=n_samples)
        
        # State |1>: Center depends on decay time tau ~ Exp(1/t)
        # If tau > 1, no decay (center at 1). If tau < 1, center at tau.
        mask_1 = (states == 1)
        if np.any(mask_1):
            n_1 = np.sum(mask_1)
            # Sample decay times
            decay_times = np.random.exponential(scale=1.0/self.t, size=n_1)
            # Integration center: effective signal is min(1, tau) because readout stops at t=1
            centers_1 = np.minimum(1.0, decay_times)
            z_values[mask_1] = centers_1 + np.random.normal(0, self.sigma, size=n_1)
            
        # State |2>: Gaussian at 2, decay rate 2t
        mask_2 = (states == 2)
        if np.any(mask_2):
            n_2 = np.sum(mask_2)
            decay_times = np.random.exponential(scale=1.0/(2*self.t), size=n_2)
            # Center shifts: Starts at 2. Decays toward 1.
            # Effective center = 1 + min(1, tau)
            centers_2 = 1.0 + np.minimum(1.0, decay_times)
            z_values[mask_2] = centers_2 + np.random.normal(0, self.sigma, size=n_2)
            
        return z_values.reshape(measurements.shape)

    def calculate_posteriors(self, z_values):
        """
        Converts 'z' signals to probabilities (Decoder Inputs) using Bayes Rule.
        """
        p0 = self._pdf_0(z_values)
        p1 = self._pdf_1(z_values)
        p2 = self._pdf_2(z_values)
        
        total_prob = (self.w0 * p0) + (self.w1 * p1) + (self.w2 * p2)
        
        # Posterior for |2> (Leakage Probability) [cite: 610]
        post_2 = (self.w2 * p2) / total_prob
        
        # Posterior for |1> conditioned on not |2> (Signal Probability) [cite: 608, 612]
        # prob(1 | !2) = prob(1) / (prob(0) + prob(1))
        # Note: In the paper (eq 612), this is (w1*p1) / (w0*p0 + w1*p1)
        denom_no_leak = (self.w0 * p0) + (self.w1 * p1)
        # Avoid division by zero
        denom_no_leak = np.maximum(denom_no_leak, 1e-15)
        post_1 = (self.w1 * p1) / denom_no_leak
        
        return post_1, post_2

    def soft_xor(self, p_curr, p_prev):
        """
        Combines two measurement probabilities into a detection event probability.
        Formula: q = p_n(1 - p_{n-1}) + (1 - p_n)p_{n-1} [cite: 635]
        """
        return p_curr * (1 - p_prev) + (1 - p_curr) * p_prev