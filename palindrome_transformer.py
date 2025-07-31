import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, LayerNormalization, Add,
    MultiHeadAttention, Bidirectional, GRU, Conv1D, GlobalMaxPooling1D,
    Concatenate, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.utils import register_keras_serializable

# Architecture constants - Scaled up for ~1M parameters
NUM_HEADS = 8  # Increased from 2
GRU_UNITS = 512  # Increased from 128
FEEDFORWARD_MULTIPLIER = 4  # Increased from 2
CLASSIFIER_UNITS = [256, 128, 64]  # Increased from [64, 32]
MIN_VAL_LOSS = 0.3


@register_keras_serializable(package="Custom", name="LearnedPositionalEncoding")
class LearnedPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, input_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_len = input_len
        self.embed = Embedding(input_dim=input_len, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.input_len)
        pos = self.embed(positions)
        return tf.broadcast_to(pos, [tf.shape(x)[0], self.input_len, tf.shape(pos)[-1]])

    def get_config(self):
        return {"input_len": self.input_len, "embed_dim": self.embed.output_dim}


# Scaled up embed_dim from 128 to 512 for ~1M parameters
def build_hybrid_model(input_len=12, vocab_size=27, embed_dim=512, gamma=2.0):
    """Build a hybrid model with ~1M parameters - NO SYMMETRY FEATURE."""
    inp = Input(shape=(input_len,), name="input")

    # Embedding + Positional Encoding
    x = Embedding(vocab_size, embed_dim, mask_zero=True, name="embedding")(inp)
    pos = LearnedPositionalEncoding(
        input_len, embed_dim, name="positional_encoding")(inp)
    x = Add(name="add_embedding")([x, pos])

    # Multiple Transformer encoder blocks for deeper processing
    for i in range(3):  # Increased from 1 to 3 layers
        attn = MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=embed_dim, dropout=0.1, name=f"multi_head_attention_{i}")(x, x)
        x = Add(name=f"add_attn_{i}")([x, attn])
        x = LayerNormalization(name=f"layer_normalization_1_{i}")(x)

        # Feedforward + norm
        ff = Dense(embed_dim * FEEDFORWARD_MULTIPLIER,
                   activation="relu", name=f"dense_ff1_{i}")(x)
        ff = Dense(embed_dim, name=f"dense_ff2_{i}")(ff)
        x = Add(name=f"add_ff_{i}")([x, ff])
        x = LayerNormalization(name=f"layer_normalization_2_{i}")(x)

    # BiGRU layer with more units
    x = Bidirectional(GRU(GRU_UNITS, return_sequences=False, name="bigru"))(x)
    x = Dropout(0.3, name="dropout_after_gru")(x)

    # NO SYMMETRY FEATURE - removed the concatenation

    # Enhanced classifier with more layers
    x = Dense(CLASSIFIER_UNITS[0], activation="relu", name="dense1_1")(x)
    x = Dropout(0.25, name="dropout_after_dense1_1")(x)
    x = Dense(CLASSIFIER_UNITS[1], activation="relu", name="dense1_2")(x)
    x = Dropout(0.2, name="dropout_after_dense1_2")(x)
    x = Dense(CLASSIFIER_UNITS[2], activation="relu", name="dense1_3")(x)
    x = Dropout(0.15, name="dropout_after_dense1_3")(x)
    out = Dense(1, activation="sigmoid", name="dense_output")(x)

    model = Model(inputs=inp, outputs=out)  # Only one input now

    # Use fixed learning rate - start with BinaryCrossentropy for stability
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", AUC(name="auc"), Precision(
            name="precision"), Recall(name="recall")],
    )

    return model


# Keep the original model for comparison
def build_hybrid_model_with_symmetry(input_len=12, vocab_size=27, embed_dim=512, gamma=2.0):
    """Build a hybrid model with ~1M parameters - WITH SYMMETRY FEATURE."""
    inp = Input(shape=(input_len,), name="input")
    symmetry_inp = Input(shape=(1,), name="symmetry_input")

    # Embedding + Positional Encoding
    x = Embedding(vocab_size, embed_dim, mask_zero=True, name="embedding")(inp)
    pos = LearnedPositionalEncoding(
        input_len, embed_dim, name="positional_encoding")(inp)
    x = Add(name="add_embedding")([x, pos])

    # Multiple Transformer encoder blocks for deeper processing
    for i in range(3):  # Increased from 1 to 3 layers
        attn = MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=embed_dim, dropout=0.1, name=f"multi_head_attention_{i}")(x, x)
        x = Add(name=f"add_attn_{i}")([x, attn])
        x = LayerNormalization(name=f"layer_normalization_1_{i}")(x)

        # Feedforward + norm
        ff = Dense(embed_dim * FEEDFORWARD_MULTIPLIER,
                   activation="relu", name=f"dense_ff1_{i}")(x)
        ff = Dense(embed_dim, name=f"dense_ff2_{i}")(ff)
        x = Add(name=f"add_ff_{i}")([x, ff])
        x = LayerNormalization(name=f"layer_normalization_2_{i}")(x)

    # BiGRU layer with more units
    x = Bidirectional(GRU(GRU_UNITS, return_sequences=False, name="bigru"))(x)
    x = Dropout(0.3, name="dropout_after_gru")(x)

    # Combine with symmetry feature - ensure consistent types
    symmetry_inp_float = tf.cast(symmetry_inp, tf.float32)
    x = Concatenate(name="concat_after_gru")([x, symmetry_inp_float])

    # Enhanced classifier with more layers
    x = Dense(CLASSIFIER_UNITS[0], activation="relu", name="dense1_1")(x)
    x = Dropout(0.25, name="dropout_after_dense1_1")(x)
    x = Dense(CLASSIFIER_UNITS[1], activation="relu", name="dense1_2")(x)
    x = Dropout(0.2, name="dropout_after_dense1_2")(x)
    x = Dense(CLASSIFIER_UNITS[2], activation="relu", name="dense1_3")(x)
    x = Dropout(0.15, name="dropout_after_dense1_3")(x)
    out = Dense(1, activation="sigmoid", name="dense_output")(x)

    model = Model(inputs=[inp, symmetry_inp], outputs=out)

    # Use fixed learning rate - start with BinaryCrossentropy for stability
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", AUC(name="auc"), Precision(
            name="precision"), Recall(name="recall")],
    )

    return model
