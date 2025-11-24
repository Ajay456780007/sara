import keras
from keras import layers
from keras.utils import plot_model


# Define a MultiHeadSelfAttention layer (simplified for demonstration)
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.proj_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def call(self, inputs):
        # Simplified attention logic for visualization purposes
        q = self.query_dense(inputs)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)
        # In a real implementation, you'd perform scaled dot-product attention
        # and combine heads. For plot_model, the layer connections are key.
        return self.combine_heads(v)  # Placeholder for attention output


# Define the Transformer Encoder Block
def transformer_encoder_block(inputs, embed_dim, num_heads, ff_dim, rate=0.1):
    attn_output = MultiHeadSelfAttention(embed_dim, num_heads)(inputs)
    attn_output = layers.Dropout(rate)(attn_output)
    norm1_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = layers.Dense(ff_dim, activation="relu")(norm1_output)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(rate)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(norm1_output + ffn_output)


# Build a simple Transformer Encoder model
def build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_blocks):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, embed_dim, num_heads, ff_dim)
    outputs = layers.Dense(1, activation="sigmoid")(x[:, -1, :])  # Example output layer
    return keras.Model(inputs=inputs, outputs=outputs)


# Model parameters
input_shape = (20, 32)  # Sequence length, embedding dimension
embed_dim = 32
num_heads = 2
ff_dim = 64
num_blocks = 2

# Create the model
model = build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_blocks)

# Plot the model
plot_model(model, to_file='transformer_encoder.png', show_shapes=True, show_layer_names=True)
