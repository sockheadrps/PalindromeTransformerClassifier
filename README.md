# Palindrome Detection with Reflection-Integrated Neural Network

A palindrome detection system that uses a custom character reflection score alongside traditional neural network features. The model achieves perfect accuracy by combining text processing with symmetry analysis.

![Test Results](pally.png)

## What it does

This project trains a neural network to detect palindromes (words/phrases that read the same forwards and backwards) using two inputs:

- The text itself (encoded as numbers)
- A custom "reflection score" that measures character-by-character symmetry

The reflection score is calculated by comparing the left and right halves of the text, giving the model a direct measure of symmetry to work with.


## Project Structure

```
pdrome/
├── models/
│   └── model_with_reflection_fixed.keras    # Trained model
├── static/
│   └── index.html                           # Web interface
├── multiword_data_gen.py                    # Data generation & reflection score
├── simple_reflection_fix.py                 # Model training
├── fast_comprehensive_test.py               # Testing suite
├── palindrome_router.py                     # API endpoints
├── app_fastapi.py                          # Web server
└── requirements.txt                         # Dependencies
```

## How the Reflection Score Works

The reflection score measures how symmetric a string is by:

1. Splitting the text into left and right halves
2. Converting each character to a number (-1 to 1)
3. Comparing the left half with the reversed right half
4. Returning a score from 0 (no symmetry) to 1 (perfect symmetry)

```python
def char_reflection_score(s):
    # Clean text (remove spaces/punctuation)
    s = ''.join(c for c in s.lower() if c.isalpha())

    # Split into halves
    n = len(s)
    half = n // 2

    if n % 2 == 0:  # Even length
        left = s[:half]
        right = s[half:]
    else:  # Odd length
        left = s[:half]
        right = s[half+1:]  # Skip middle character

    right = right[::-1]  # Reverse right half

    # Convert characters to numbers
    def embed(c):
        return (ord(c) - 97 - 13) / 13

    # Compare vectors
    vec1 = [embed(c) for c in left]
    vec2 = [embed(c) for c in right]

    diff = [abs(a - b) for a, b in zip(vec1, vec2)]
    return 1.0 - sum(diff) / len(diff)
```

## Performance

The model achieves perfect accuracy across all test cases:

- **Overall**: 100.0% (1000/1000)
- **Palindromes**: 100.0% (500/500)
- **Non-palindromes**: 100.0% (500/500)
- **Length range**: 3-50 characters
- **Model size**: ~4.4MB

## Quick Start

### Setup

```bash
python -m venv .venv310
.venv310\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Test the model

```bash
python fast_comprehensive_test.py
```

### Start the web interface

```bash
python app_fastapi.py
```

Then visit http://localhost:8000/monitor

### Train a new model

```bash
python simple_reflection_fix.py
```

## Web Interface

The FastAPI server provides:

- **Real-time testing**: Try any text and see results instantly
- **Visualizations**: See how the model analyzes your input
- **Model status**: Check if the model is loaded and its performance
- **Test results**: View comprehensive test statistics

Visit http://localhost:8000/monitor after starting the server.

## Model Architecture

The neural network has two inputs:

1. **Text encoding**: 50-dimensional vector representing the text
2. **Reflection score**: Single number measuring symmetry

The model uses:

- Bidirectional GRU layers for text processing
- Dense layers for combining text and reflection features
- Binary output (palindrome/not palindrome)
- Adam optimizer with early stopping

## Testing

The comprehensive test suite generates 1000 test cases:

- 500 palindromes, 500 non-palindromes
- Evenly distributed across lengths 3-50
- Includes spaces and punctuation (automatically cleaned)
- Reports accuracy by length ranges and palindrome type

## Dependencies

- TensorFlow 2.x
- NumPy
- Matplotlib
- FastAPI
- Uvicorn

## License

MIT License
