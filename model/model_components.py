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
        
        # Pre-Norm: normalize before attention for better gradient flow
        x_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        attn_out = attention_layer(query=x_norm, key=x_norm, value=x_norm, mask=mask)
        x = x + attn_out
        return x


# --- 3. FFN BLOCK ---
class FFNBlock(hk.Module):
    """Standard feed-forward network block (transformer-style). No auxiliary loss."""

    def __init__(self, model_dim, expansion_factor=4, name=None):
        super().__init__(name=name)
        self.model_dim = model_dim
        self.intermediate_dim = model_dim * expansion_factor

    def __call__(self, x):
        # x: (batch, num_stabilizers, model_dim)
        out = hk.Linear(self.intermediate_dim)(x)
        out = jax.nn.relu(out)
        out = hk.Linear(self.model_dim)(out)
        return out


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


# --- 5. MAIN TRANSFORMER ---
class SyndromeTransformer(hk.Module):
    def __init__(self, model_dim=64, distance=3, name=None):
        super().__init__(name=name)
        self.model_dim = model_dim
        self.distance = distance

    def __call__(self, x):
        # Input 'x' is already model_dim from the RNN_core.

        # Instantiate Layers
        attention_layer = SelfAttentionBlock(num_heads=4, key_size=16, model_size=self.model_dim)
        conv2D = SurfaceCodeConv(filters=self.model_dim, kernel_size=3, stride=1, distance=self.distance)

        # 1. Attention (pre-norm is inside SelfAttentionBlock)
        x = attention_layer(x)

        # 2. FFN block with pre-norm and residual
        ffn = FFNBlock(model_dim=self.model_dim, expansion_factor=4)
        x_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = x + ffn(x_norm)

        # 3. Convolution with pre-norm and residual
        x_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = x + conv2D(x_norm)

        return x


# --- 6. RNN CORE ---
class RNN_core(hk.Module):
    def __init__(self, mixing_mult, distance=3, num_layers=3, model_dim=64, name=None):
        super().__init__(name=name)
        self.mixing_mult = mixing_mult
        self.distance = distance
        self.num_layers = num_layers
        self.model_dim = model_dim

    def __call__(self, s, decoder_state):
        # GRU-style gated state update for stable gradient flow over many rounds
        # Gate decides how much new input to incorporate vs carrying forward state
        combined = jnp.concatenate([s, decoder_state], axis=-1)  # (B, S, 2*model_dim)
        gate = jax.nn.sigmoid(hk.Linear(self.model_dim, name="gate")(combined))  # (B, S, model_dim)
        
        # Gated mixing: gate=1 → use new input, gate=0 → carry old state
        x = gate * s + (1 - gate) * decoder_state

        # Deep processing: Stack of N SyndromeTransformer layers
        for i in range(self.num_layers):
            x = SyndromeTransformer(
                model_dim=self.model_dim,
                distance=self.distance,
                name=f"transformer_{i+1}"
            )(x)
        return x

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


        

        