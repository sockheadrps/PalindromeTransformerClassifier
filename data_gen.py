import random
import string
import numpy as np  
from dotenv import load_dotenv
import os
import random
from collections import Counter

load_dotenv()
MAXLEN = int(os.getenv("MAXLEN"))


def is_palindrome(s):
    return s == s[::-1]

def generate_string(length, palindrome=False):
    if palindrome:
        half = ''.join(random.choices(string.ascii_lowercase, k=length // 2))
        return half + half[::-1] if length % 2 == 0 else half + random.choice(string.ascii_lowercase) + half[::-1]
    else:
        s = ''.join(random.choices(string.ascii_lowercase, k=length))
        while is_palindrome(s):
            s = ''.join(random.choices(string.ascii_lowercase, k=length))
        return s


def make_near_palindrome(p):
    i = random.randint(0, len(p)-1)
    c = random.choice([c for c in string.ascii_lowercase if c != p[i]])
    return p[:i] + c + p[i+1:]


def generate_dataset(n=1000, maxlen=MAXLEN):
    data = []

    # Load real palindromes from file
    try:
        with open("palindromes.txt", "r") as f:
            real_palindromes = [line.strip().lower() for line in f if 3 <= len(line.strip()) <= maxlen]
    except FileNotFoundError:
        print("palindromes.txt not found! Proceeding with only generated data.")
        real_palindromes = []

    # Add a chunk of real palindromes first
    for _ in range(n // 4):  # 25% real palindromes
        if real_palindromes:
            s = random.choice(real_palindromes)
            data.append((s, 1))

    # Generate the rest
    for _ in range(n - len(data)):
        length = random.randint(3, maxlen)
        if random.random() > 0.5:
            s = generate_string(length, palindrome=True)
            label = 1
        else:
            s = generate_string(length, palindrome=False)
            label = 0
        data.append((s, label))

    with open("not_palindromes.txt") as f:
        hard_negatives = [line.strip().lower() for line in f if 3 <= len(line.strip()) <= maxlen]
        for _ in range(100):
            data.append((random.choice(hard_negatives), 0))
    

    # Add near-palindromes negative training data
    for _ in range(100):
        if real_palindromes:
            p = random.choice(real_palindromes)
            near = make_near_palindrome(p)
            data.append((near, 0))  # label as not palindrome

            
    random.shuffle(data)
    label_counts = Counter(label for _, label in data)
    print(f"Dataset label balance: {label_counts}")
    return data


def preprocess(data, maxlen=MAXLEN):
    import string
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    alphabet = list(string.ascii_lowercase)
    char_to_index = {c: i+1 for i, c in enumerate(alphabet)}  # index 0 = padding

    def encode(word):
        return [char_to_index.get(c, 0) for c in word]

    X = [encode(word) for word, _ in data]
    y = [label for _, label in data]
    X = pad_sequences(X, maxlen=maxlen, padding='post')
    y = np.array(y)  # ← Convert y to NumPy array here
    return X, y


