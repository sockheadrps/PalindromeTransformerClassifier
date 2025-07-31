import random
import string
import numpy as np
import os
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAXLEN = 50  # Hardcoded value for multi-word

# Alphabetic vocabulary only (no spaces or punctuation)
EXTENDED_CHARS = string.ascii_lowercase
CHAR_TO_INDEX = {c: i + 1 for i, c in enumerate(EXTENDED_CHARS)}
INDEX_TO_CHAR = {v: k for k, v in CHAR_TO_INDEX.items()}
CHAR_TO_INDEX['<PAD>'] = 0
INDEX_TO_CHAR[0] = '<PAD>'


def char_reflection_score(s):
    """Calculate symmetry score based on character reflection."""
    s = ''.join(c for c in s.lower() if c in string.ascii_lowercase)
    n = len(s)
    half = n // 2

    if n % 2 == 0:
        left = s[:half]
        right = s[half:]
    else:
        left = s[:half]
        right = s[half+1:]

    right = right[::-1]

    def embed(c):
        return (ord(c) - 97 - 13) / 13  # zero-centered

    vec1 = np.fromiter(map(embed, left), dtype=float)
    vec2 = np.fromiter(map(embed, right), dtype=float)

    diff = vec1 - vec2
    return 1.0 - np.mean(np.abs(diff))  # 1.0 = perfect symmetry


def char_reflection_score_detailed(s):
    """Calculate symmetry score with detailed vector information."""
    s = ''.join(c for c in s.lower() if c in string.ascii_lowercase)
    n = len(s)
    half = n // 2

    if n % 2 == 0:
        left = s[:half]
        right = s[half:]
        center_char = None
    else:
        left = s[:half]
        right = s[half+1:]
        center_char = s[half]

    right_reversed = right[::-1]

    def embed(c):
        return (ord(c) - 97 - 13) / 13  # zero-centered

    def embed_raw(c):
        return ord(c) - 97  # raw character position (a=0, b=1, etc.)

    vec1 = np.fromiter(map(embed, left), dtype=float)
    vec2 = np.fromiter(map(embed, right_reversed), dtype=float)

    vec1_raw = np.fromiter(map(embed_raw, left), dtype=int)
    vec2_raw = np.fromiter(map(embed_raw, right_reversed), dtype=int)

    diff = vec1 - vec2
    mean_abs_diff = np.mean(np.abs(diff))
    score = 1.0 - mean_abs_diff  # 1.0 = perfect symmetry

    # Create character mappings for better visualization
    left_chars = list(left)
    right_chars = list(right_reversed)

    left_mapping = [{'char': char, 'value': float(val), 'raw_value': int(raw_val)}
                    for char, val, raw_val in zip(left_chars, vec1, vec1_raw)]
    right_mapping = [{'char': char, 'value': float(val), 'raw_value': int(raw_val)}
                     for char, val, raw_val in zip(right_chars, vec2, vec2_raw)]
    diff_mapping = [{'left_char': lc, 'right_char': rc, 'difference': float(d)}
                    for lc, rc, d in zip(left_chars, right_chars, diff)]

    return {
        'score': score,
        'left_half': left,
        'right_half': right,
        'right_reversed': right_reversed,
        'center_char': center_char,
        'left_vector': vec1.tolist(),
        'right_vector': vec2.tolist(),
        'difference_vector': diff.tolist(),
        'mean_abs_difference': mean_abs_diff,
        'normalization_formula': '(ord(c) - 97 - 13) / 13',
        'score_formula': '1.0 - mean_abs_difference',
        'left_mapping': left_mapping,
        'right_mapping': right_mapping,
        'difference_mapping': diff_mapping
    }


def is_multiword_palindrome(text):
    """Check if text is a palindrome ignoring spaces and punctuation."""
    # Remove spaces and punctuation, convert to lowercase
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1] and len(cleaned) > 0


def generate_structured_palindrome_pattern(min_len=3, max_len=7):
    """Generate structured palindrome patterns with palindrome words."""
    half_len = random.randint(min_len // 2, max_len // 2)
    words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 6)))
             for _ in range(half_len)]

    # Make each word a palindrome
    words = [w if w == w[::-1] else w + w[::-1] for w in words]

    mirrored = words[::-1]
    return ' '.join(words + mirrored)


def generate_multiword_palindrome(max_words=5, max_word_len=8):
    """Generate a multi-word palindrome."""
    if max_words == 1:
        # Single word palindrome
        word_len = random.randint(3, max_word_len)
        half = ''.join(random.choices(string.ascii_lowercase, k=word_len // 2))
        if word_len % 2 == 0:
            word = half + half[::-1]
        else:
            mid = random.choice(string.ascii_lowercase)
            word = half + mid + half[::-1]
        return word

    # Use structured pattern for multi-word palindromes
    if random.random() < 0.4:  # 70% chance to use structured pattern
        return generate_structured_palindrome_pattern(2, max_words)

    # Original method as fallback
    words = []
    for i in range(max_words // 2):
        word_len = random.randint(3, max_word_len)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)

    # Mirror the words for palindrome structure
    mirrored_words = words[::-1]

    # Add middle word if odd number of words
    if max_words % 2 == 1:
        middle_word_len = random.randint(3, max_word_len)
        middle_word = ''.join(random.choices(
            string.ascii_lowercase, k=middle_word_len))
        # Make middle word itself a palindrome
        half = middle_word[:middle_word_len // 2]
        if middle_word_len % 2 == 0:
            middle_word = half + half[::-1]
        else:
            mid = middle_word[middle_word_len // 2]
            middle_word = half + mid + half[::-1]
        words.append(middle_word)

    words.extend(mirrored_words)

    # Join without spaces (alphabetic only)
    result = ''.join(words)

    # Verify it's actually a palindrome
    if not is_multiword_palindrome(result):
        # If not, create a simple palindrome
        word_len = random.randint(3, max_word_len)
        half = ''.join(random.choices(string.ascii_lowercase, k=word_len // 2))
        if word_len % 2 == 0:
            result = half + half[::-1]
        else:
            mid = random.choice(string.ascii_lowercase)
            result = half + mid + half[::-1]

    return result


def generate_non_multiword_palindrome(max_words=5, max_word_len=8):
    """Generate a non-palindrome multi-word text."""
    if max_words == 1:
        # Single word non-palindrome
        word_len = random.randint(3, max_word_len)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        while word == word[::-1]:  # Ensure it's not a palindrome
            word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        return word

    # Multi-word non-palindrome
    words = []
    for _ in range(max_words):
        word_len = random.randint(3, max_word_len)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)

    # Ensure it's not a palindrome by checking the cleaned version
    text = ''.join(words)
    if is_multiword_palindrome(text):
        # If it accidentally became a palindrome, change one word
        idx = random.randint(0, len(words) - 1)
        new_word = ''.join(random.choices(
            string.ascii_lowercase, k=len(words[idx])))
        while new_word == new_word[::-1]:  # Ensure new word isn't palindrome
            new_word = ''.join(random.choices(
                string.ascii_lowercase, k=len(words[idx])))
        words[idx] = new_word
        text = ''.join(words)

    return text


def encode_multiword(text, maxlen=MAXLEN):
    """Encode multi-word text to indices."""
    # Preprocess text to remove whitespace and special characters (same as reflection score)
    cleaned_text = ''.join(c.lower()
                           for c in text if c in string.ascii_lowercase)
    encoded = [CHAR_TO_INDEX.get(c, 0) for c in cleaned_text]
    return pad_sequences([encoded], maxlen=maxlen, padding='post')[0]


def decode_multiword(encoded):
    """Decode indices back to text."""
    return ''.join(INDEX_TO_CHAR.get(i, '<UNK>') for i in encoded if i != 0)


def generate_challenging_multiword_palindrome():
    """Generate more challenging multi-word palindromes with better patterns."""
    patterns = [
        # Simple 2-word patterns
        ["level", "level"],
        ["radar", "radar"],
        ["deed", "deed"],

        # 4-word patterns
        ["step", "on", "no", "pets"],
        ["live", "not", "on", "evil"],
        ["do", "geese", "see", "god"],

        # 5-word patterns
        ["race", "fast", "safe", "car"],
        ["never", "odd", "or", "even"],

        # 7-word patterns
        ["a", "man", "a", "plan", "a", "canal", "panama"],
    ]

    # Filter out patterns that are not actually palindromes
    valid_patterns = []
    for pattern in patterns:
        text = ' '.join(pattern)
        if is_multiword_palindrome(text):
            valid_patterns.append(pattern)
        else:
            print(f"⚠️ Removing non-palindrome pattern: {' '.join(pattern)}")

    if not valid_patterns:
        # Fallback to simple palindromes if no valid patterns
        valid_patterns = [
            ["level", "level"],
            ["radar", "radar"],
            ["deed", "deed"],
            ["step", "on", "no", "pets"],
            ["live", "not", "on", "evil"],
            ["do", "geese", "see", "god"],
            ["race", "fast", "safe", "car"],
            ["never", "odd", "or", "even"],
            ["a", "man", "a", "plan", "a", "canal", "panama"],
        ]

    pattern = random.choice(valid_patterns)
    return ' '.join(pattern)


def generate_realistic_palindrome_pattern():
    """Generate realistic palindrome patterns that look more natural."""
    # Common palindrome word patterns
    palindrome_words = [
        "level", "radar", "deed", "noon", "civic", "rotor", "kayak",
        "anna", "bob", "dad", "mom", "pop", "wow", "eye", "eve"
    ]

    # Generate 2-4 palindrome words
    num_words = random.randint(2, 4)
    words = random.sample(palindrome_words, num_words)

    # Mirror the sequence
    mirrored = words[::-1]
    return ' '.join(words + mirrored)


def generate_balanced_multiword_dataset(n_samples=3000, maxlen=MAXLEN, max_total_len=None, multiword_ratio=0.7):
    """Generate balanced dataset of multi-word palindromes and non-palindromes with increased multi-word focus."""
    palindromes = []
    non_palindromes = []

    # Calculate how many multi-word vs single-word examples to generate
    multiword_count = int(n_samples * multiword_ratio)
    singleword_count = n_samples - multiword_count

    # Generate palindromes - keep generating until we have enough
    while len(palindromes) < n_samples // 2:
        if len(palindromes) < multiword_count // 2:  # First portion: multi-word palindromes
            choice = random.random()
            if choice < 0.2:  # 20% chance for challenging examples
                palindrome = generate_challenging_multiword_palindrome()
            elif choice < 0.5:  # 30% chance for realistic patterns
                palindrome = generate_realistic_palindrome_pattern()
            else:
                # 50% chance for structured patterns
                max_words = random.randint(2, 6)
                palindrome = generate_multiword_palindrome(max_words)
        else:  # Second portion: single-word palindromes
            max_words = 1
            palindrome = generate_multiword_palindrome(max_words)

        if len(palindrome) <= maxlen and (max_total_len is None or len(palindrome) <= max_total_len):
            # Validate that it's actually a palindrome
            if is_multiword_palindrome(palindrome):
                palindromes.append(palindrome)
            else:
                print(
                    f"⚠️ Warning: Generated non-palindrome labeled as palindrome: '{palindrome}'")

    # Generate non-palindromes - keep generating until we have enough
    while len(non_palindromes) < n_samples // 2:
        # First portion: multi-word non-palindromes
        if len(non_palindromes) < multiword_count // 2:
            max_words = random.randint(2, 6)  # Force multi-word (2-6 words)
        else:  # Second portion: single-word non-palindromes
            max_words = 1

        non_palindrome = generate_non_multiword_palindrome(max_words)
        if len(non_palindrome) <= maxlen and (max_total_len is None or len(non_palindrome) <= max_total_len):
            # Validate that it's actually not a palindrome
            if not is_multiword_palindrome(non_palindrome):
                non_palindromes.append(non_palindrome)
            else:
                print(
                    f"⚠️ Warning: Generated palindrome labeled as non-palindrome: '{non_palindrome}'")

    # Ensure we have exactly equal numbers
    min_count = min(len(palindromes), len(non_palindromes))
    palindromes = palindromes[:min_count]
    non_palindromes = non_palindromes[:min_count]

    # Combine and shuffle
    data = [(text, 1) for text in palindromes] + [(text, 0)
                                                  for text in non_palindromes]
    random.shuffle(data)

    texts, labels = zip(*data)

    # Encode texts
    encoded_texts = np.array(
        [encode_multiword(text, maxlen) for text in texts])
    labels = np.array(labels)

    # Calculate symmetry scores for each text
    symmetry_scores = np.array([char_reflection_score(text) for text in texts])

    # Reshape symmetry scores to be a column
    symmetry_scores = symmetry_scores.reshape(-1, 1)

    return encoded_texts, labels, symmetry_scores


def create_hard_multiword_examples():
    """Create challenging multi-word palindrome examples."""
    # Initial lists of examples
    initial_palindromes = [
        "a man a plan a canal panama",
        "do geese see god",
        "never odd or even",
        "madam i'm adam",
        "a roza upala na lapu azora",
        "step on no pets",
        "was it a car or a cat i saw",
        "no x in nixon",
        "race fast safe car",
        "live not on evil",
    ]

    initial_non_palindromes = [
        "this is not a palindrome",
        "hello world",
        "machine learning is fun",
        "python programming",
        "artificial intelligence",
        "deep learning models",
        "neural networks",
        "computer science",
        "data structures",
        "algorithms and complexity",
        "software engineering",
        "web development",
        "mobile applications",
        "cloud computing",
        "cybersecurity",
        "database systems",
        "operating systems",
        "computer architecture",
        "distributed systems",
        "human computer interaction",
        "almost looks symmetrical",
        "madam I'm a dog",
        "step on no pet",
        "was it a dog or a cat i saw",
    ]

    # Validate palindromes - only keep actual palindromes
    hard_palindromes = []
    for text in initial_palindromes:
        if is_multiword_palindrome(text):
            hard_palindromes.append(text)
        else:
            print(
                f"⚠️ Warning: '{text}' is not actually a palindrome, removing from hard examples")

    # Validate non-palindromes - only keep actual non-palindromes
    hard_non_palindromes = []
    for text in initial_non_palindromes:
        if not is_multiword_palindrome(text):
            hard_non_palindromes.append(text)
        else:
            print(
                f"⚠️ Warning: '{text}' is actually a palindrome, removing from non-palindrome examples")

    print(
        f"✅ Validated hard examples: {len(hard_palindromes)} palindromes, {len(hard_non_palindromes)} non-palindromes")

    return hard_palindromes, hard_non_palindromes


if __name__ == "__main__":
    # Test the functions
    print("Testing multi-word palindrome generation:")

    # Test palindrome generation
    for _ in range(5):
        palindrome = generate_multiword_palindrome(max_words=3)
        print(
            f"Palindrome: '{palindrome}' -> {is_multiword_palindrome(palindrome)}")

    print("\nTesting non-palindrome generation:")
    for _ in range(5):
        non_palindrome = generate_non_multiword_palindrome(max_words=3)
        print(
            f"Non-palindrome: '{non_palindrome}' -> {is_multiword_palindrome(non_palindrome)}")

    # Test encoding/decoding
    test_text = "a man a plan"
    encoded = encode_multiword(test_text)
    decoded = decode_multiword(encoded)
    print(f"\nEncoding test: '{test_text}' -> {encoded} -> '{decoded}'")
