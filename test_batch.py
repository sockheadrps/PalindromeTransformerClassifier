import string
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import BinaryFocalCrossentropy
from dotenv import load_dotenv
import os

load_dotenv()
MAXLEN = int(os.getenv("MAXLEN", 10))

# Build char-to-index map
char_to_index = {c: i + 1 for i, c in enumerate(string.ascii_lowercase)}

def encode_word(word, maxlen=MAXLEN):
    encoded = [[char_to_index.get(c, 0) for c in word.lower()]]
    return pad_sequences(encoded, maxlen=maxlen, padding='post')

def load_words(path):
    with open(path, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]

def predict_and_print(model, test_set):
    print(f"{'Word':<12} {'Expected':<12} {'Prediction':<22} {'Confidence'}")
    print("-" * 60)

    for word, label in test_set:
        padded = encode_word(word)
        confidence = model.predict(padded)[0][0]
        predicted_label = int(confidence > 0.5)

        verdict = (
            "✅ Correct" if predicted_label == label else "❌ Wrong"
        )
        prediction_text = (
            "✅ Palindrome" if confidence > 0.7 else
            "❌ Not a palindrome"
        )
        expected = "Palindrome" if label == 1 else "Not a palindrome"

        print(f"{word:<12} {expected:<12} {prediction_text:<22} ({confidence:.2f}) {verdict}")

if __name__ == "__main__":
    model = load_model("palindrome_model.keras", custom_objects={
        "focal_loss": BinaryFocalCrossentropy(gamma=2.5)
    })

    palindromes = load_words("palindromes.txt")
    not_palindromes = load_words("not_palindromes.txt")

    sample = []
    sample += [(w, 1) for w in random.sample(palindromes, 15)]
    sample += [(w, 0) for w in random.sample(not_palindromes, 15)]
    random.shuffle(sample)

    predict_and_print(model, sample)
