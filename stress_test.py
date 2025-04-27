import string
import random
import os
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_gen import generate_string
from model import LearnedPositionalEncoding
from tensorflow.keras.losses import BinaryFocalCrossentropy


load_dotenv()
MAXLEN = int(os.getenv("MAXLEN", 10))
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
MODEL_PATH = os.getenv("LOAD_MODEL_PATH") + "\model.keras"
# MODEL_PATH = os.getenv("LOAD_MODEL_PATH") + "\polished_model.keras"




char_to_index = {c: i + 1 for i, c in enumerate(string.ascii_lowercase)}

def encode_word(word, maxlen=MAXLEN):
    encoded = [[char_to_index.get(c, 0) for c in word.lower()]]
    return pad_sequences(encoded, maxlen=maxlen, padding='post')

def load_words(path):
    with open(path, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]

def generate_test_set(count=100, maxlen=MAXLEN):
    test = []
    for _ in range(count):
        length = random.randint(3, maxlen)
        test.append((generate_string(length, palindrome=True), 1))
        test.append((generate_string(length, palindrome=False), 0))
    random.shuffle(test)
    return test

def predict_and_print(model, test_set, threshold=THRESHOLD):
    print(f"{'Word':<15} {'Expected':<18} {'Prediction':<22} {'Confidence'}")
    print("-" * 70)

    for word, label in test_set:
        confidence = model.predict(encode_word(word))[0][0]
        pred = int(confidence > threshold)

        expected = ["Not a palindrome", "Palindrome"][label]
        predicted = ["❌ Not a palindrome", "✅ Palindrome"][pred]
        verdict = "✅ Correct" if pred == label else "❌ Wrong"

        print(f"{word:<15} {expected:<18} {predicted:<22} ({confidence:.2f}) {verdict}")

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH,
    custom_objects={
        "LearnedPositionalEncoding": LearnedPositionalEncoding,
        "BinaryFocalCrossentropy": BinaryFocalCrossentropy,
    }
)

    palindromes = load_words("palindromes.txt")
    not_palindromes = load_words("not_palindromes.txt")

    # Mix of real + hard negatives
    sample = [(w, 1) for w in random.sample(palindromes, 25)]
    sample += [(w, 0) for w in random.sample(not_palindromes, 25)]

    # Add synthetic palindromes and anti-palindromes
    generated = generate_test_set(count=10)

    full_test = sample + generated
    random.shuffle(full_test)

    predict_and_print(model, full_test)
