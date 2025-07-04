import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, LayerNormalization, Add,
    MultiHeadAttention, Bidirectional, GRU
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.saving import register_keras_serializable

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

def build_hybrid_model(input_len=12, vocab_size=27, embed_dim=64, gamma=2.0):
    inp = Input(shape=(input_len,), name="input")

    # Embedding + Positional Encoding
    x = Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    pos = LearnedPositionalEncoding(input_len, embed_dim)(inp)
    x = Add()([x, pos])

    # Transformer encoder block
    attn = MultiHeadAttention(num_heads=2, key_dim=embed_dim, dropout=0.1)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    # Feedforward + norm
    ff = Dense(embed_dim * 2, activation="relu")(x)
    ff = Dense(embed_dim)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)

    # BiGRU layer for symmetry/time sensitivity
    x = Bidirectional(GRU(128, return_sequences=False))(x)
    x = Dropout(0.3)(x)

    # Dense classifier
    x = Dense(64, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation="relu", kernel_initializer="he_uniform")(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    # Learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1500,
        decay_rate=0.97,
        staircase=False,
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=BinaryFocalCrossentropy(gamma=gamma),
        metrics=["accuracy", AUC(name="auc")],
    )

    return model
