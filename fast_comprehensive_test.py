#!/usr/bin/env python3
"""
Fast comprehensive test for the reflection-integrated model with 1000 palindromes 
evenly distributed across lengths 3-50.
"""

import numpy as np
import tensorflow as tf
import string
import random
from multiword_data_gen import encode_multiword


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

    vec1 = np.array([embed(c) for c in left])
    vec2 = np.array([embed(c) for c in right])

    diff = vec1 - vec2
    return 1.0 - np.mean(np.abs(diff))


def load_model():
    """Load the reflection-integrated model."""
    try:
        model = tf.keras.models.load_model(
            "models/model_with_reflection_fixed.keras")
        print("✅ Reflection-integrated model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def generate_simple_palindrome(length):
    """Generate a simple palindrome of given length."""
    chars = string.ascii_lowercase
    half = length // 2

    if length % 2 == 0:
        # Even length
        half_str = ''.join(random.choice(chars) for _ in range(half))
        return half_str + half_str[::-1]
    else:
        # Odd length
        half_str = ''.join(random.choice(chars) for _ in range(half))
        middle = random.choice(chars)
        return half_str + middle + half_str[::-1]


def generate_non_palindrome(length):
    """Generate a non-palindrome of given length."""
    chars = string.ascii_lowercase
    result = ''.join(random.choice(chars) for _ in range(length))

    # Ensure it's not a palindrome
    if result == result[::-1]:
        # Change the last character
        result = result[:-1] + \
            random.choice([c for c in chars if c != result[-1]])

    return result


def generate_test_cases_fast():
    """Generate 1000 test cases quickly."""
    print("🎯 Generating 1000 test cases quickly...")

    test_cases = []
    total_cases = 1000
    length_range = 50 - 3 + 1  # 3 to 50 inclusive
    cases_per_length = total_cases // length_range
    remaining_cases = total_cases % length_range

    # Track cases per length for detailed reporting
    length_distribution = {}

    for length in range(3, 51):
        # Calculate cases for this length
        cases_this_length = cases_per_length
        if length <= 3 + remaining_cases - 1:
            cases_this_length += 1

        length_distribution[length] = cases_this_length

        # Generate palindromes
        for _ in range(cases_this_length // 2):
            palindrome = generate_simple_palindrome(length)
            # Add spaces randomly
            if random.random() < 0.3:
                positions = random.sample(
                    range(1, len(palindrome)), min(2, len(palindrome)-1))
                palindrome_list = list(palindrome)
                for pos in positions:
                    palindrome_list.insert(pos, ' ')
                palindrome = ''.join(palindrome_list)
            test_cases.append((palindrome, True, length))

        # Generate non-palindromes
        for _ in range(cases_this_length // 2):
            non_palindrome = generate_non_palindrome(length)
            # Add spaces randomly
            if random.random() < 0.3:
                positions = random.sample(
                    range(1, len(non_palindrome)), min(2, len(non_palindrome)-1))
                non_palindrome_list = list(non_palindrome)
                for pos in positions:
                    non_palindrome_list.insert(pos, ' ')
                non_palindrome = ''.join(non_palindrome_list)
            test_cases.append((non_palindrome, False, length))

    # Shuffle test cases
    random.shuffle(test_cases)

    print(f"✅ Generated {len(test_cases)} test cases")

    # Print detailed length distribution
    print("\n📊 Test Case Distribution by Length:")
    print("-" * 40)
    total_reported = 0
    for length in sorted(length_distribution.keys()):
        count = length_distribution[length]
        total_reported += count
        print(f"   Length {length:2d}: {count:3d} cases")
    print(f"   {'Total':>8}: {total_reported:3d} cases")

    return test_cases


def test_model_fast(model, test_cases):
    """Test the model on comprehensive test cases."""
    print("🧪 Running comprehensive test...")

    results = []
    length_stats = {}

    for i, (text, expected, length) in enumerate(test_cases):
        if i % 100 == 0:
            print(f"   Testing case {i+1}/{len(test_cases)}...")

        # Encode text
        encoded = encode_multiword(text, maxlen=50)
        encoded = np.expand_dims(encoded, 0)

        # Calculate reflection score
        reflection_score = char_reflection_score(text)
        reflection_input = np.array([[reflection_score]])

        # Get prediction
        prediction = model.predict(
            [encoded, reflection_input], verbose=0)[0][0]
        is_palindrome = prediction > 0.5

        # Store result
        result = {
            'text': text,
            'expected': expected,
            'prediction': prediction,
            'is_palindrome': is_palindrome,
            'correct': expected == is_palindrome,
            'length': length,
            'reflection_score': reflection_score
        }
        results.append(result)

        # Update length statistics
        if length not in length_stats:
            length_stats[length] = {'correct': 0, 'total': 0}
        length_stats[length]['total'] += 1
        if result['correct']:
            length_stats[length]['correct'] += 1

    return results, length_stats


def analyze_results_fast(results, length_stats):
    """Analyze and display comprehensive test results."""
    print("\n📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)

    # Overall statistics
    total_correct = sum(1 for r in results if r['correct'])
    total_cases = len(results)
    overall_accuracy = total_correct / total_cases

    print(
        f"📈 Overall Accuracy: {overall_accuracy:.1%} ({total_correct}/{total_cases})")

    # Accuracy by palindrome type
    palindromes = [r for r in results if r['expected']]
    non_palindromes = [r for r in results if not r['expected']]

    palindrome_accuracy = sum(
        1 for r in palindromes if r['correct']) / len(palindromes) if palindromes else 0
    non_palindrome_accuracy = sum(
        1 for r in non_palindromes if r['correct']) / len(non_palindromes) if non_palindromes else 0

    print(
        f"📈 Palindrome Accuracy: {palindrome_accuracy:.1%} ({sum(1 for r in palindromes if r['correct'])}/{len(palindromes)})")
    print(
        f"📈 Non-palindrome Accuracy: {non_palindrome_accuracy:.1%} ({sum(1 for r in non_palindromes if r['correct'])}/{len(non_palindromes)})")

    # Accuracy by length ranges
    print(f"\n📊 Accuracy by Length Ranges:")
    print("-" * 40)

    length_ranges = [
        (3, 10, "Short (3-10 chars)"),
        (11, 20, "Medium (11-20 chars)"),
        (21, 30, "Long (21-30 chars)"),
        (31, 50, "Very Long (31-50 chars)")
    ]

    for min_len, max_len, label in length_ranges:
        range_results = [r for r in results if min_len <=
                         r['length'] <= max_len]
        if range_results:
            correct = sum(1 for r in range_results if r['correct'])
            accuracy = correct / len(range_results)
            print(f"   {label}: {accuracy:.1%} ({correct}/{len(range_results)})")

    # Find incorrect predictions
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print(f"\n❌ Sample Incorrect Predictions (showing first 10):")
        for r in incorrect[:10]:
            print(
                f"   '{r['text']}' (len {r['length']}) -> {r['prediction']:.4f} (expected: {'palindrome' if r['expected'] else 'not palindrome'})")
        if len(incorrect) > 10:
            print(f"   ... and {len(incorrect) - 10} more")

    return overall_accuracy


def main():
    """Main function for comprehensive testing."""
    print("🎯 Fast Comprehensive Test for Reflection-Integrated Model")
    print("=" * 60)

    # Load model
    model = load_model()
    if model is None:
        return

    # Generate test cases
    test_cases = generate_test_cases_fast()

    # Run comprehensive test
    results, length_stats = test_model_fast(model, test_cases)

    # Analyze results
    accuracy = analyze_results_fast(results, length_stats)

    print(f"\n🎉 Comprehensive test completed!")
    print(f"📊 Final Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
