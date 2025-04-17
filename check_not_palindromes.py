import re

def is_palindrome(s):
    return s == s[::-1]

def clean(word):
    return re.sub(r'[^a-z]', '', word.lower())

def check_file(filename="not_palindromes.txt"):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]

    found = False
    for line in lines:
        cleaned = clean(line)
        if len(cleaned) > 2 and is_palindrome(cleaned):
            print(f"⚠️  Found palindrome in not_palindromes.txt: '{line}'")
            found = True

    if not found:
        print("✅ All entries in not_palindromes.txt are valid non-palindromes.")

if __name__ == "__main__":
    check_file()
