import random
import string
import numpy as np
from dotenv import load_dotenv
import os
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

load_dotenv()
MAXLEN = int(os.getenv("MAXLEN"))


def decode_batch(batch, index_to_char=None, pad_char="_"):
    if index_to_char is None:
        index_to_char = {i + 1: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    return [
        "".join(index_to_char.get(char_id, pad_char) for char_id in sequence)
        for sequence in batch
    ]


def is_palindrome(s):
    return s == s[::-1]


def generate_string(length, palindrome=False, seen=None):
    seen = seen or set()
    max_tries = 20
    for _ in range(max_tries):
        if palindrome:
            half = ''.join(random.choices(string.ascii_lowercase, k=length // 2))
            s = half + half[::-1] if length % 2 == 0 else half + random.choice(string.ascii_lowercase) + half[::-1]
        else:
            s = ''.join(random.choices(string.ascii_lowercase, k=length))
            while is_palindrome(s):
                s = ''.join(random.choices(string.ascii_lowercase, k=length))
        if s not in seen:
            seen.add(s)
            return s
    return None


def make_double_letter_non_palindrome(length):
    s = []
    for _ in range(length // 2):
        c = random.choice(string.ascii_lowercase)
        s.extend([c, c])  # force a double letter
    if len(s) > length:
        s = s[:length]
    random.shuffle(s)
    return ''.join(s)



def make_near_palindrome(p):
    i = random.randint(0, len(p) - 1)
    wrong_char = random.choice([c for c in string.ascii_lowercase if c != p[i]])
    return p[:i] + wrong_char + p[i+1:]

def make_very_near_palindrome(p, n_changes=2):
    p = list(p)
    indices = random.sample(range(len(p)), n_changes)
    for idx in indices:
        original = p[idx]
        choices = [c for c in string.ascii_lowercase if c != original]
        p[idx] = random.choice(choices)
    return ''.join(p)

def make_partial_mirror_non_palindrome(length):
    half_len = length // 2
    first_half = ''.join(random.choices(string.ascii_lowercase, k=half_len))
    mirrored = first_half[::-1]

    s = first_half + mirrored

    # Break the outer ends so it isn't truly symmetric
    s = random.choice(string.ascii_lowercase) + s[1:-1] + random.choice(string.ascii_lowercase)

    return s


def make_symetrical_non_palindrome(length):
    """Generate a symmetric-looking non-palindrome."""
    half = ''.join(random.choices(string.ascii_lowercase, k=length // 2))
    flipped = list(half[::-1])

    # Flip one letter in the second half to break symmetry
    if flipped:
        idx = random.randint(0, len(flipped) - 1)
        original = flipped[idx]
        options = [c for c in string.ascii_lowercase if c != original]
        flipped[idx] = random.choice(options)

    if length % 2 == 0:
        s = half + ''.join(flipped)
    else:
        mid = random.choice(string.ascii_lowercase)
        s = half + mid + ''.join(flipped)

    return s


import random
from collections import Counter

def generate_balanced_dataset(n_samples=3000, maxlen=MAXLEN, stage="full"):
    data = []

    # Load real palindromes and hard negatives
    try:
        with open("palindromes.txt", "r") as f:
            real_palindromes = [line.strip().lower() for line in f if 1 <= len(line.strip()) <= maxlen]
    except FileNotFoundError:
        print("palindromes.txt not found! Proceeding with only generated palindromes.")
        real_palindromes = []

    try:
        with open("not_palindromes.txt", "r") as f:
            hard_negatives = [line.strip().lower() for line in f if 1 <= len(line.strip()) <= maxlen]
    except FileNotFoundError:
        print("not_palindromes.txt not found! Proceeding without hard negatives.")
        hard_negatives = []

    def weighted_length():
        return random.choices(
            [3, 4, 5, random.randint(6, maxlen)],
            weights=[4, 3, 2, 1]
        )[0]

    seen = set()

    # --- Main generation logic varies by stage ---
    while len(data) < n_samples:
        length = weighted_length()

        if stage == "easy":
            if random.random() < 0.85:
                s = generate_string(length, palindrome=True, seen=seen)
                label = 1
            else:
                s = generate_string(length, palindrome=False, seen=seen)
                label = 0
        elif stage == "medium":
            if random.random() < 0.7:
                s = generate_string(length, palindrome=True, seen=seen)
                label = 1
            else:
                s = generate_string(length, palindrome=False, seen=seen)
                label = 0
        elif stage == "hard1":
            if random.random() < 0.6:
                s = generate_string(length, palindrome=True, seen=seen)
                label = 1
            else:
                s = generate_string(length, palindrome=False, seen=seen)
                label = 0
        elif stage == "hard2":
            if random.random() < 0.5:
                s = generate_string(length, palindrome=True, seen=seen)
                label = 1
            else:
                s = generate_string(length, palindrome=False, seen=seen)
                label = 0
        else:  # full
            if random.random() < 0.5:
                s = generate_string(length, palindrome=True, seen=seen)
                label = 1
            else:
                s = generate_string(length, palindrome=False, seen=seen)
                label = 0

        if s:
            data.append((s, label))

    # --- Only full stage includes hard negative boosting ---
    if stage == "full":
        # Add near-palindromes (contrastive examples)
        for p in real_palindromes:
            if 3 <= len(p) <= maxlen:
                data.append((p, 1))                         # Real palindrome
                data.append((make_near_palindrome(p), 0))   # Broken palindrome

        # Add very near-palindromes
        for p in random.sample(real_palindromes, min(100, len(real_palindromes))):
            data.append((make_very_near_palindrome(p), 0))

        # Add symmetrical non-palindromes
        for _ in range(100):
            length = random.randint(3, maxlen)
            s = make_symetrical_non_palindrome(length)
            if not is_palindrome(s):
                data.append((s, 0))

        # Add double-letter non-palindromes
        for _ in range(100):
            length = random.randint(3, maxlen)
            s = make_double_letter_non_palindrome(length)
            if not is_palindrome(s):
                data.append((s, 0))

        # Add partial mirror structures
        for _ in range(100):
            length = random.randint(3, maxlen)
            s = make_partial_mirror_non_palindrome(length)
            data.append((s, 0))

        # Add hard negatives if available
        if hard_negatives:
            for s in random.sample(hard_negatives, min(100, len(hard_negatives))):
                data.append((s, 0))

    # --- Inject fake symmetry into hard2 (new) ---
    if stage == "hard2":
        for _ in range(50):
            length = random.randint(3, maxlen)
            s = make_symetrical_non_palindrome(length)
            if not is_palindrome(s):
                data.append((s, 0))

    # Final prep
    data = list(set(data))
    random.shuffle(data)
    label_counts = Counter(label for _, label in data)
    print(f"Dataset label balance: {label_counts}")

    return data


def preprocess(data, maxlen=MAXLEN):
    alphabet = list(string.ascii_lowercase)
    char_to_index = {c: i+1 for i, c in enumerate(alphabet)}  # 0 = padding

    def encode(word):
        return [char_to_index.get(c, 0) for c in word]

    X = [encode(word) for word, _ in data]
    y = [label for _, label in data]
    X = pad_sequences(X, maxlen=maxlen, padding='post')
    y = np.array(y)
    return X, y

def load_data():
    train_data = generate_balanced_dataset(n_samples=50000, maxlen=12)
    x_train_raw, y_train = zip(*train_data)

    val_data = generate_balanced_dataset(n_samples=10000, maxlen=12)
    x_val_raw, y_val = zip(*val_data)

    return list(x_train_raw), list(y_train), list(x_val_raw), list(y_val)

if __name__ == "__main__":
    p = generate_string(10, palindrome=True)
    print(is_palindrome(p), p)
    p = generate_string(10, palindrome=False)
    print(is_palindrome(p), p)


    
    # double letter non palindrome\
    print(f"make_double_letter_non_palindrome is used to make {make_double_letter_non_palindrome(10)} \n")
    print("the model seemed to often think any word with double letters was a palindrome")

    # near palindrome
    print(f"make_near_palindrome is used to make {make_near_palindrome('racecar')}, with just one letter changed")

    # very near palindrome
    print(f"make_very_near_palindrome is used to make {make_very_near_palindrome('racecar', n_changes=2)}, with multiple letters changed")

    # partial mirror non palindrome
    print(f"make_partial_mirror_non_palindrome Builds a perfect mirror, then breaks only the outer letters {make_partial_mirror_non_palindrome(10)}")

    # symmetric non palindrome  
    print(f"make_symetrical_non_palindrome Builds a mirror image, then breaks one random letter in the second half {make_symetrical_non_palindrome(10)}")
    