import jax 
import jax.numpy as jnp
import haiku as hk

# --- 1. STABILIZER EMBEDDER ---
# Keeps the raw syndrome bit processing separate
class StabilizerEmbedder(hk.Module):
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, x):
        # x shape: (batch, num_stabilizers, input_features)
        # e.g. (batch, 8, 4) for d=3 with 4 channels: Post_1, Event_1, Post_2, Event_2
        return hk.Linear(self.output_size)(x)


# --- 2. SELF ATTENTION BLOCK ---
class SelfAttentionBlock(hk.Module):
    def __init__(self, num_heads, key_size, model_size, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
    
    def __call__(self, x, mask=None):
        attention_layer = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            model_size=self.model_size,
            w_init=hk.initializers.VarianceScaling(1.0)
        )
        
        # Pre-Norm architecture (often more stable for deep transformers)
        # But Post-Norm (your style) is also fine. Keeping your style:
        attn_out = attention_layer(query=x, key=x, value=x, mask=mask)
        x = x + attn_out
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        return x


# --- 3. MOE LAYER ---
class MixtureOfExpertsLayer(hk.Module):
    """MoE with load-balancing auxiliary loss to prevent expert collapse."""

    def __init__(self, num_experts, expert_dim, name=None):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.expert_dim = expert_dim

    def __call__(self, x):
        model_dim = x.shape[-1]

        # Router
        gate_logits = hk.Linear(self.num_experts)(x)
        gate_weights = jax.nn.softmax(gate_logits, axis=-1)

        # Load-balancing auxiliary loss: penalize imbalanced expert usage.
        # mean_gate_e = fraction of "tokens" routed to expert e.
        # aux_loss = num_experts * sum_e(mean_gate_e^2); minimized when uniform (1/N each).
        mean_gate = jnp.mean(gate_weights, axis=(0, 1))  # (num_experts,)
        aux_loss = self.num_experts * jnp.sum(mean_gate ** 2)

        # Experts
        expert_outputs = []
        for i in range(self.num_experts):
            expert_net = hk.Sequential([
                hk.Linear(self.expert_dim),
                jax.nn.relu,
                hk.Linear(model_dim)
            ])
            expert_outputs.append(expert_net(x))

        stacked_experts = jnp.stack(expert_outputs, axis=0)

        # Weighted Combination
        # gate: [Batch, Stab, Exp], stack: [Exp, Batch, Stab, Dim]
        combined_output = jnp.einsum('bse,ebsd->bsd', gate_weights, stacked_experts)

        return combined_output, aux_loss


# --- 4. SURFACE CODE CONV ---
class SurfaceCodeConv(hk.Module):
    def __init__(self, filters, kernel_size, stride, distance=3, padding="SAME", name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.distance = distance

        # Build the checkerboard stabilizer map for the given distance.
        # For a rotated surface code of distance d, stabilizers are placed on
        # a (d+1) x (d+1) grid in a checkerboard pattern, yielding d^2 - 1 stabilizers.
        grid_size = distance + 1
        rows_list = []
        cols_list = []
        for r in range(grid_size):
            for c in range(grid_size):
                # Checkerboard: pick cells where (r + c) is odd
                if (r + c) % 2 == 1:
                    rows_list.append(r)
                    cols_list.append(c)
        self.rows = jnp.array(rows_list)
        self.cols = jnp.array(cols_list)
        self.grid_size = grid_size

    def __call__(self, x):
        batch, num_stabilizers, features = x.shape

        # Learned Padding
        padding_vector = hk.get_parameter(
            "padding_vector", 
            shape=(1, 1, 1, features), 
            init=hk.initializers.TruncatedNormal()
        )
        
        # Scatter -> Conv -> Gather
        grid = jnp.tile(padding_vector, (batch, self.grid_size, self.grid_size, 1))
        grid = grid.at[:, self.rows, self.cols, :].set(x)

        conv_layer = hk.Conv2D(
            output_channels=self.filters,
            kernel_shape=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            rate=1, 
            with_bias=True
        )
        grid_out = conv_layer(grid)
        out = grid_out[:, self.rows, self.cols, :]

        return out


# --- 5. MAIN TRANSFORMER (The Fix is Here) ---
class SyndromeTransformer(hk.Module):
    def __init__(
        self,
        model_dim=64,
        distance=3,
        use_moe=False,
        ffn_hidden_dim=128,
        moe_num_experts=4,
        moe_expert_dim=128,
        name=None,
    ):
        super().__init__(name=name)
        self.model_dim = model_dim
        self.distance = distance
        self.use_moe = use_moe
        self.ffn_hidden_dim = ffn_hidden_dim
        self.moe_num_experts = moe_num_experts
        self.moe_expert_dim = moe_expert_dim

    def __call__(self, x):
        attention_layer = SelfAttentionBlock(num_heads=4, key_size=16, model_size=self.model_dim)
        conv2D = SurfaceCodeConv(filters=self.model_dim, kernel_size=3, stride=1, distance=self.distance)

        # 1. Attention
        x = attention_layer(x)

        # 2. Feed-forward block (dense by default; optional MoE)
        if self.use_moe:
            moe = MixtureOfExpertsLayer(
                num_experts=self.moe_num_experts,
                expert_dim=self.moe_expert_dim,
            )
            x_ffn, aux_loss = moe(x)
        else:
            dense_block = hk.Sequential([
                hk.Linear(self.ffn_hidden_dim),
                jax.nn.relu,
                hk.Linear(self.model_dim),
            ])
            x_ffn = dense_block(x)
            aux_loss = jnp.array(0.0, dtype=x.dtype)
        x = x + x_ffn
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        # 3. Convolution (Spatial Context)
        x_conv = conv2D(x)
        x = x + x_conv
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        return x, aux_loss


# --- 6. RNN CORE ---
class RNN_core(hk.Module):
    def __init__(
        self,
        mixing_mult,
        distance=3,
        num_layers=3,
        model_dim=64,
        use_moe=False,
        ffn_hidden_dim=128,
        moe_num_experts=4,
        moe_expert_dim=128,
        name=None,
    ):
        super().__init__(name=name)
        self.mixing_mult = mixing_mult
        self.distance = distance
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.use_moe = use_moe
        self.ffn_hidden_dim = ffn_hidden_dim
        self.moe_num_experts = moe_num_experts
        self.moe_expert_dim = moe_expert_dim

    def __call__(self, s, decoder_state):
        # Fixed paper-style scaled sum mixing.
        x = self.mixing_mult * (s + decoder_state)

        # Deep processing: Stack of N SyndromeTransformer layers
        aux_total = 0.0
        for i in range(self.num_layers):
            x, aux = SyndromeTransformer(
                model_dim=self.model_dim,
                distance=self.distance,
                use_moe=self.use_moe,
                ffn_hidden_dim=self.ffn_hidden_dim,
                moe_num_experts=self.moe_num_experts,
                moe_expert_dim=self.moe_expert_dim,
                name=f"transformer_{i+1}"
            )(x)
            aux_total = aux_total + aux

        return x, aux_total / self.num_layers

class ReadoutHead(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, decoder_state):
        # decoder_state shape: (Batch, 8, 64)
        
        # 1. Aggregate information from all 8 stabilizers
        # You can use mean, max, or just flatten. 
        # AlphaQubit often uses a weighted sum or attention, 
        # but mean is perfect for d=3.
        pooled = jnp.mean(decoder_state, axis=1)  # Shape: (Batch, 64)
        
        # 2. Final Classification
        # We want 1 output (logit) representing P(Logical Error)
        logits = hk.Linear(1)(pooled) 
        
        return logits # Shape: (Batch, 1)


        

        