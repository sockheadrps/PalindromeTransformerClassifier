from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LayerNormalization



def build_model(input_len=10, vocab_size=27, gamma=2.5):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=16, mask_zero=True),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=True)),
        LayerNormalization(),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_initializer='he_uniform'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # binary classification
    ])

    lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0015,
    decay_steps=1000,       # every 1000 steps...
    decay_rate=0.98,        # multiply lr by 0.96
    staircase=False          # set to False for smooth decay
)

    opt = Adam(learning_rate=lr_schedule)

    # https://keras.io/api/losses/probabilistic_losses/#binaryfocalcrossentropy-class
    model.compile(
        optimizer=opt,
        loss=BinaryFocalCrossentropy(gamma=gamma),
        metrics=['accuracy', AUC(name="auc")]
    )

    return model
