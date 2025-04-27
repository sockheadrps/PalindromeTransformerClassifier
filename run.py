from data_gen import generate_balanced_dataset, preprocess
from model import build_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import os
import json
from datetime import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau
import dotenv


dotenv.load_dotenv()
MAXLEN = int(os.getenv("MAXLEN", 12))
N_SAMPLES = int(os.getenv("N_SAMPLES", 70000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 50))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.2))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 8))


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
save_dir = f"saved_models/model_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

def encode(word, maxlen=MAXLEN):
    char_to_index = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    encoded = [[char_to_index.get(c, 0) for c in word.lower()]]
    return pad_sequences(encoded, maxlen=maxlen, padding='post')

# 2. Data
data = generate_balanced_dataset(N_SAMPLES)
X, y = preprocess(data, maxlen=MAXLEN)

# 3. Model
model = build_model(input_len=MAXLEN)

# 4. Save model architecture early (not weights yet)
model_json = model.to_json()
with open(os.path.join(save_dir, "model_structure.json"), "w") as f:
    f.write(model_json)

# 5. Save training config
train_config = {
    "timestamp": timestamp,
    "n_samples": N_SAMPLES,
    "maxlen": MAXLEN,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "validation_split": VALIDATION_SPLIT,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
}
with open(os.path.join(save_dir, "train_config.json"), "w") as f:
    json.dump(train_config, f, indent=4)

print(f"Model structure and config saved to {save_dir}")


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
# 6. Training
history = model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    ],
)

# 7. Save final trained model
model.save(os.path.join(save_dir, "model.keras"))

# 8. Post-train check
print("\n==== Testing Post-Train Confidence ====")
for word in ["racecar", "tacocat", "noon", "apple", "banana"]:
    pred = model.predict(encode(word))[0][0]
    print(f"{word}: {pred:.4f}")

# 9. Plot training curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "training_plot.png"))
plt.show()
