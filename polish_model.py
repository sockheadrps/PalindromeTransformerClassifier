import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.metrics import AUC
from model import LearnedPositionalEncoding
from data_gen import preprocess, load_data 
import os
import dotenv
import numpy as np


dotenv.load_dotenv()
MAXLEN = int(os.getenv("MAXLEN", 12))
MODEL_PATH = os.getenv("LOAD_MODEL_PATH")

N_SAMPLES = int(os.getenv("N_SAMPLES", 70000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 50))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.2))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 8))

def preprocess_words(data, maxlen=MAXLEN):
    import string
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    alphabet = list(string.ascii_lowercase)
    char_to_index = {c: i+1 for i, c in enumerate(alphabet)}  # 0 = padding

    def encode(word):
        return [char_to_index.get(c, 0) for c in word]

    X = [encode(word) for word in data]
    X = pad_sequences(X, maxlen=maxlen, padding='post')
    return np.array(X)

# Load full trained model
model = tf.keras.models.load_model(
    f"{MODEL_PATH}/model.keras",
    custom_objects={
        "LearnedPositionalEncoding": LearnedPositionalEncoding,  # Needed because you have custom layer
        "BinaryFocalCrossentropy": BinaryFocalCrossentropy,
    }
)

# Freeze all except Dense layers
for layer in model.layers:
    if not isinstance(layer, tf.keras.layers.Dense):
        layer.trainable = False

# Recompile with tiny learning rate
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=opt,
    loss=BinaryFocalCrossentropy(gamma=2.0),
    metrics=['accuracy', AUC(name="auc")]
)

# (Re)load data
x_train_raw, y_train, x_val_raw, y_val = load_data()

# PREPROCESS
x_train = preprocess_words(x_train_raw)
x_val = preprocess_words(x_val_raw)

# Convert labels to numpy arrays
y_train = np.array(y_train)
y_val = np.array(y_val)

# x_train and x_val are numeric tensors

# Train lightly
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop],
)

# Save polished model
model.save(f"{MODEL_PATH}/polished_model.keras")
