import os
import random
import string
import time
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.metrics import confusion_matrix
from data_gen import generate_string
from model import LearnedPositionalEncoding

# Load environment variables
load_dotenv()
MAXLEN = int(os.getenv("MAXLEN", 12))
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
# MODEL_PATH = os.path.join(os.getenv("LOAD_MODEL_PATH"), "model.keras")
MODEL_PATH = "polished_retrained_model.keras"


# Char index mapping
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
    correct = 0
    incorrect_cases = []
    y_true = []
    y_pred = []

    for word, label in test_set:
        start = time.time()
        confidence = model.predict(encode_word(word), verbose=0)[0][0]
        end = time.time()

        pred = int(confidence > threshold)
        y_true.append(label)
        y_pred.append(pred)

        if pred == label:
            correct += 1
        else:
            incorrect_cases.append({
                "word": word,
                "expected": "Palindrome" if label == 1 else "Not a palindrome",
                "predicted": "Palindrome" if pred == 1 else "Not a palindrome",
                "confidence": confidence,
                "time": end - start,
            })

    total = len(test_set)
    incorrect = total - correct
    acc = correct / total

    print(f"\n🎯 Accuracy: {correct}/{total} = {acc:.2%}")
    print(f"✅ Correct: {correct}")
    print(f"❌ Incorrect: {incorrect}")

    if incorrect_cases:
        print("\n🧾 Misclassified Examples:")
        print(f"{'Word':<15} {'Expected':<18} {'Predicted':<18} {'Confidence':<12} {'Time (s)'}")
        print("-" * 80)
        for case in incorrect_cases:
            print(f"{case['word']:<15} {case['expected']:<18} {case['predicted']:<18} ({case['confidence']:.2f})     {case['time']:.3f}s")

    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_true, y_pred))



if __name__ == "__main__":
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "LearnedPositionalEncoding": LearnedPositionalEncoding,
            "BinaryFocalCrossentropy": BinaryFocalCrossentropy,
        }
    )

    # Load from files
    palindromes = [w for w in load_words("palindromes.txt") if len(w) <= MAXLEN]
    not_palindromes = [w for w in load_words("not_palindromes.txt") if len(w) <= MAXLEN]

    sample = [(w, 1) for w in random.sample(palindromes, 200)]
    sample += [(w, 0) for w in random.sample(not_palindromes, 200)]

    generated = generate_test_set(count=200)

    # Final combined test set
    full_test = sample + generated
    random.shuffle(full_test)

    predict_and_print(model, full_test)
