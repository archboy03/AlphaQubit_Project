import haiku as hk

from .model_components import RNN_core, StabilizerEmbedder, ReadoutHead

class CycleArchitecture(hk.Module):
    def __init__(self, mixing_mult=0.7, output_size=64, distance=3, name=None):
        super().__init__(name=name)
        self.mixing_mult = mixing_mult
        self.output_size = output_size
        self.distance = distance

    def __call__(self, new_checks, d_old):
        # RNN logic
        embedder = StabilizerEmbedder(self.output_size)
        s = embedder(new_checks)
        
        rnn_core = RNN_core(self.mixing_mult, distance=self.distance)
        d_new = rnn_core(s, d_old)
        
        # Final readout network
        readout = ReadoutHead()
        final_logit = readout(d_new)
        
        # Return BOTH the state (for the next round) AND the prediction (for loss)
        return d_new, final_logit