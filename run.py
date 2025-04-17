from data_gen import generate_dataset, preprocess
from model import build_model
from predict import predict_word
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


MAXLEN = int(os.getenv("MAXLEN"))


def encode(word, maxlen=MAXLEN):
    char_to_index = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    encoded = [[char_to_index.get(c, 0) for c in word.lower()]]
    return pad_sequences(encoded, maxlen=maxlen, padding='post')

data = generate_dataset(3000)
X, y = preprocess(data, maxlen=MAXLEN)
model = build_model(input_len=MAXLEN)


history = model.fit(
    X, y,
    epochs=75,
    batch_size=32,
    validation_split=0.2,
)


# 3. Post-train confidence check
print("==== Testing Post-Train Confidence ====")
for word in ["racecar", "tacocat", "noon", "apple", "banana"]:
    pred = model.predict(encode(word))[0][0]
    print(f"{word}: {pred:.4f}")

# 4. Try predictions using helper
print("\nPredictions:")
for word in ["racecar", "apple", "noon", "banana", "tacocat"]:
    result, pred = predict_word(model, word)
    status = "✅ Palindrome" if pred > 0.70 else "❌ Not a palindrome"
    print(f"{word}: {status} ({pred:.2f})")

# 5. Save model
model.save("palindrome_model.keras")

# 6. Plot results
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
plt.show()
