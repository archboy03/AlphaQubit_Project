import haiku as hk

from .model_components import RNN_core, StabilizerEmbedder, ReadoutHead

class CycleArchitecture(hk.Module):
    def __init__(
        self,
        mixing_mult=0.7,
        output_size=64,
        distance=3,
        num_layers=3,
        use_moe=False,
        ffn_hidden_dim=128,
        moe_num_experts=4,
        moe_expert_dim=128,
        name=None,
    ):
        super().__init__(name=name)
        self.mixing_mult = mixing_mult
        self.output_size = output_size
        self.distance = distance
        self.num_layers = num_layers
        self.use_moe = use_moe
        self.ffn_hidden_dim = ffn_hidden_dim
        self.moe_num_experts = moe_num_experts
        self.moe_expert_dim = moe_expert_dim

    def __call__(self, new_checks, d_old):
        # RNN logic
        embedder = StabilizerEmbedder(self.output_size)
        s = embedder(new_checks)

        rnn_core = RNN_core(
            self.mixing_mult,
            distance=self.distance,
            num_layers=self.num_layers,
            model_dim=self.output_size,
            use_moe=self.use_moe,
            ffn_hidden_dim=self.ffn_hidden_dim,
            moe_num_experts=self.moe_num_experts,
            moe_expert_dim=self.moe_expert_dim,
        )
        d_new, aux_loss = rnn_core(s, d_old)

        # Final readout network
        readout = ReadoutHead()
        final_logit = readout(d_new)

        # Return state, prediction, and MoE load-balancing auxiliary loss
        return d_new, final_logit, aux_loss
