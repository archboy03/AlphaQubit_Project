import haiku as hk

from .model_components import RNN_core, StabilizerEmbedder, ReadoutHead

class CycleArchitecture(hk.Module):
    def __init__(self, mixing_mult=0.7, output_size=64, distance=3, num_layers=3, name=None):
        super().__init__(name=name)
        self.mixing_mult = mixing_mult
        self.output_size = output_size
        self.distance = distance
        self.num_layers = num_layers

    def __call__(self, new_checks, d_old):
        # RNN logic
        embedder = StabilizerEmbedder(self.output_size)
        s = embedder(new_checks)

        rnn_core = RNN_core(
            self.mixing_mult,
            distance=self.distance,
            num_layers=self.num_layers,
            model_dim=self.output_size,
        )
        d_new, aux_loss = rnn_core(s, d_old)

        # Final readout network
        readout = ReadoutHead()
        final_logit = readout(d_new)

        # Return state, prediction, and MoE load-balancing auxiliary loss
        return d_new, final_logit, aux_loss
