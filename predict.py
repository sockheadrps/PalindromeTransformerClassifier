import string
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryFocalCrossentropy
from dotenv import load_dotenv
import os

load_dotenv()
MAXLEN = int(os.getenv("MAXLEN"))


def clean_input(word):
    return re.sub(r'[^a-z]', '', word.lower())

def predict_word(model, word, maxlen=MAXLEN):
    char_to_index = {c: i+1 for i, c in enumerate(string.ascii_lowercase)}
    encoded = [[char_to_index.get(c, 0) for c in word]]
    padded = pad_sequences(encoded, maxlen=maxlen, padding='post')
    print("Encoded padded input:", padded)
    pred = model.predict(padded)[0][0]
    return pred > 0.5, pred

if __name__ == "__main__":
    import sys
    model = load_model("palindrome_model.keras", custom_objects={"focal_loss": BinaryFocalCrossentropy()})
    word = clean_input(sys.argv[1] if len(sys.argv) > 1 else "racecar")
    result, pred = predict_word(model, word)
    status = (
        "✅ Palindrome" if pred > 0.7 else
        "❌ Not a palindrome"
    )
    print(f"{word}: {status} ({pred:.2f})")
    print(f"{word}: {status} ({pred:.2f} confidence)")
