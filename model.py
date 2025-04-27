from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dropout, Dense, LayerNormalization, Add, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy  
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from keras.saving import register_keras_serializable
from tensorflow.keras.callbacks import ReduceLROnPlateau


def tile_positional_embedding(pos_emb, batch_size_tensor):
    return Lambda(lambda x: tf.tile(pos_emb, [tf.shape(x)[0], 1, 1]))(batch_size_tensor)

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
        config = super().get_config()
        config.update({
            "input_len": self.input_len,
            "embed_dim": self.embed.output_dim,
        })
        return config

def build_model(input_len=10, vocab_size=27, gamma=2.0):
    char_input = Input(shape=(input_len,), name='char_input')

    # Character embedding match the embed_dim to the embedding layer output.
    char_embedding = Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True)(char_input)

    # Positional encoding
    position_embedding = LearnedPositionalEncoding(input_len, 64)(char_input)

    # Combine both
    combined = Add()([char_embedding, position_embedding])

    # Recurrent + classifier
    x = Bidirectional(GRU(128, return_sequences=True))(combined)
    x = Bidirectional(GRU(64, return_sequences=True, recurrent_dropout=0.15))(x)
    x = Dropout(0.2)(x)
    x = LayerNormalization()(x)
    x = GRU(256)(x)  # Bigger final GRU
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.25)(x)  # After first dense
    x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=char_input, outputs=output)

    # Optimizer
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1500,
        decay_rate=0.975,
        staircase=False
    )

    opt = Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=opt,
        loss=BinaryFocalCrossentropy(gamma=gamma),
        metrics=['accuracy', AUC(name="auc")],
    )

    return model


# model = Sequential([
#         Embedding(input_dim=vocab_size, output_dim=32, mask_zero=True),

#         # Stacked GRUs
#         Bidirectional(GRU(32, return_sequences=True, recurrent_dropout=0.1)),
#         # Bidirectional(GRU(64, return_sequences=True, recurrent_dropout=0.1)),

#         LayerNormalization(),
#         Dropout(0.1),

#         GRU(64),  # Final GRU layer without return_sequences
#         LayerNormalization(),
    
#         Dropout(0.20),

#         # # Dense(32, activation='relu', kernel_initializer='he_uniform'),
#         Dense(32, activation='relu', kernel_regularizer=l2(1e-3)),
#         # Dropout(.15),
#         # LayerNormalization(),

#         Dense(1, activation='sigmoid')  # binary classification
#     ])

#     lr_schedule = ExponentialDecay(
#         initial_learning_rate=0.0015,
#         decay_steps=1000,
#         decay_rate=0.96,
#         staircase=False
#     )

#     opt = Adam(learning_rate=lr_schedule)

#     model.compile(
#         optimizer=opt,
#         loss=BinaryFocalCrossentropy(gamma=gamma),
#         # loss = BinaryCrossentropy(label_smoothing=0.01),
#         metrics=['accuracy', AUC(name="auc")],


#     )

#     return model