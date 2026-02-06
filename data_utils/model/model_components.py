import jax 
import jax.numpy as jnp
import haiku as hk

# --- 1. STABILIZER EMBEDDER ---
class StabilizerEmbedder(hk.Module):
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, x):
        # x shape: (batch, num_stabilizers, 1) or (batch, num_stabilizers)
        # We project the raw syndrome bit into the 64D model space.
        
        # FIX: hk.linear -> hk.Linear (Capital L)
        # FIX: Removed "+ x". You cannot add the raw input (1D) to the embedding (64D).
        return hk.Linear(self.output_size)(x)


# --- 2. SELF ATTENTION BLOCK ---
class SelfAttentionBlock(hk.Module): # FIX: hk.module -> hk.Module
    def __init__(self, num_heads, key_size, model_size, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
    
    def __call__(self, x, mask=None):
        # x shape: (num_stabilizers, model_size)

        attention_layer = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            model_size=self.model_size,
            w_init=hk.initializers.VarianceScaling(1.0) # Added init for stability
        )

        attn_out = attention_layer(query=x, key=x, value=x, mask=mask)

        # Add norm and residual connection
        # NOTE: This block already handles the "+ x"
        x = x + attn_out
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        return x


# --- 3. MOE LAYER ---
class MixtureOfExpertsLayer(hk.Module): # FIX: hk.module -> hk.Module
    def __init__(self, num_experts, expert_dim, name=None):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.expert_dim = expert_dim # FIX: Added expert_dim to __init__

    def __call__(self, x):
        # x shape: [Batch, num_stabilizers, model_dim]
        model_dim = x.shape[-1]

        # ROUTING LAYER
        gate_logits = hk.Linear(self.num_experts)(x)
        gate_weights = jax.nn.softmax(gate_logits, axis=-1)  # [Batch, 8, num_experts]

        # EXPERTS
        expert_outputs = []
        for i in range(self.num_experts):
            expert_net = hk.Sequential([
                hk.Linear(self.expert_dim),
                jax.nn.relu,
                hk.Linear(model_dim)
            ])
            expert_outputs.append(expert_net(x))

        # Stack: [num_experts, Batch, 8, 64]
        stacked_experts = jnp.stack(expert_outputs, axis=0)
        
        # COMBINE
        # Einsum notes:
        # s: batch & stabilizers (treated as one block here for broadcasting)
        # e: experts
        # d: model_dim
        # We need to be careful with Batch dimension in einsum.
        # Ideally: 'bse,ebsd->bsd' (batch, stabilizer, expert)
        # But simpler is to rely on broadcasting if shapes align.
        
        # Let's be explicit to avoid batch errors:
        # gate_weights: [Batch, Stab, Experts] -> 'bse'
        # stacked_experts: [Experts, Batch, Stab, Dim] -> 'ebsd'
        combined_output = jnp.einsum('bse,ebsd->bsd', gate_weights, stacked_experts)
        
        return combined_output


# --- 4. SURFACE CODE CONV ---
class SurfaceCodeConv(hk.Module):
    def __init__(self, filters, kernel_size, stride, padding="SAME", name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.rows = jnp.array([0, 0, 1, 1, 2, 2, 3, 3]) 
        self.cols = jnp.array([1, 3, 0, 2, 1, 3, 0, 2])

    def __call__(self, x):
        batch, num_stabilizers, features = x.shape
        grid_size = 4

        padding_vector = hk.get_parameter(
            "padding_vector", 
            shape=(1, 1, 1, features), 
            init=hk.initializers.TruncatedNormal()
        )
        
        grid = jnp.tile(padding_vector, (batch, grid_size, grid_size, 1))
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
class SyndromeTransformer(hk.Module): # FIX: hk.module -> hk.Module
    def __init__(self, model_dim=64):
        super().__init__()
        self.model_dim = model_dim

    def __call__(self, x):
        # 1. Embed raw syndrome bits first
        # Input x is likely shape (Batch, 8, 1) -> Output (Batch, 8, 64)
        embedder = StabilizerEmbedder(output_size=self.model_dim)
        x = embedder(x)

        # 2. Instantiate Layers
        # FIX: Added required arguments to constructors
        moe = MixtureOfExpertsLayer(num_experts=4, expert_dim=128)
        attention_layer = SelfAttentionBlock(num_heads=4, key_size=16, model_size=self.model_dim)
        conv2D = SurfaceCodeConv(filters=self.model_dim, kernel_size=3, stride=1)

        # 3. Apply Layers
        
        # BLOCK 1: Attention
        # FIX: Removed "+ x" because SelfAttentionBlock already does it internally
        x = attention_layer(x) 
        
        # BLOCK 2: MoE
        # MoE usually replaces the "Dense" part of a transformer.
        # It needs its own residual connection (Your class didn't have one inside).
        x_moe = moe(x)
        x = x + x_moe # Residual connection
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x) # Norm

        # BLOCK 3: Convolution (Spatial Mixing)
        x_conv = conv2D(x)
        x = x + x_conv # Residual connection
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x) # Norm

        return x


        

        