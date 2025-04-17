import string
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import BinaryFocalCrossentropy
from dotenv import load_dotenv
import os
from data_gen import generate_string  # ✅ import your generator

load_dotenv()
MAXLEN = int(os.getenv("MAXLEN", 10))

char_to_index = {c: i + 1 for i, c in enumerate(string.ascii_lowercase)}

def encode_word(word, maxlen=MAXLEN):
    encoded = [[char_to_index.get(c, 0) for c in word.lower()]]
    return pad_sequences(encoded, maxlen=maxlen, padding='post')

def generate_test_set(count=10, maxlen=MAXLEN):
    test = []
    for _ in range(count):
        length = random.randint(3, maxlen)
        test.append((generate_string(length, palindrome=True), 1))
        test.append((generate_string(length, palindrome=False), 0))
    random.shuffle(test)
    return test


def predict_and_print(model, test_set, threshold=0.7):
    for word, label in test_set:
        padded = encode_word(word)
        confidence = model.predict(padded)[0][0]
        predicted_label = int(confidence > THRESHOLD)

        expected = "Palindrome" if label == 1 else "Not a palindrome"
        prediction_text = "✅ Palindrome" if predicted_label else "❌ Not a palindrome"
        verdict = "✅ Correct" if predicted_label == label else "❌ Wrong"

        color = "\033[92m" if verdict == "✅ Correct" else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{word:<12} {expected:<16} {prediction_text:<22} ({confidence:.2f}) {verdict}{reset}")


if __name__ == "__main__":
    THRESHOLD = 0.7

    model = load_model("palindrome_model.keras", custom_objects={
        "focal_loss": BinaryFocalCrossentropy(gamma=2.5)
    })

    test_set = generate_test_set(count=10)
    predict_and_print(model, test_set)
