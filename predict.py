from model import LearnedPositionalEncoding  # <-- Must be before load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import BinaryFocalCrossentropy
from dotenv import load_dotenv
import os
import string
import sys
import tensorflow as tf

load_dotenv()
MAXLEN = int(os.getenv("MAXLEN", 10))
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
MODEL_PATH = os.getenv("LOAD_MODEL_PATH")


char_to_index = {c: i + 1 for i, c in enumerate(string.ascii_lowercase)}

def encode_word(word, maxlen=MAXLEN):
    encoded = [[char_to_index.get(c, 0) for c in word.lower()]]
    return pad_sequences(encoded, maxlen=maxlen, padding='post')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <word>")
        exit()

    # Load model with custom layer
    model = load_model(f"{MODEL_PATH}/polish_model.keras", custom_objects={
        "LearnedPositionalEncoding": LearnedPositionalEncoding,
        "binary_focal_crossentropy": BinaryFocalCrossentropy()
    })

    word = sys.argv[1].lower()
    padded = encode_word(word)
    confidence = model.predict(padded)[0][0]
    status = "✅ Palindrome" if confidence > THRESHOLD else "❌ Not a palindrome"
    print(f"{word}: {status} ({confidence:.2f})")
