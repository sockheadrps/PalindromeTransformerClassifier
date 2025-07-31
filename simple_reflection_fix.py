#!/usr/bin/env python3
"""
Simple approach: Train a new model from scratch with reflection score integration.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import string
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Embedding, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from multiword_data_gen import generate_balanced_multiword_dataset, encode_multiword, generate_multiword_palindrome, generate_non_multiword_palindrome
import traceback
import random


class PerformanceThresholdCallback(Callback):
    """Stop training when we reach excellent performance levels."""

    def __init__(self, accuracy_threshold=0.999, loss_threshold=1e-5):
        super().__init__()
        self.accuracy_threshold = accuracy_threshold
        self.loss_threshold = loss_threshold

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy', 0)
        val_loss = logs.get('val_loss', float('inf'))

        if val_accuracy >= self.accuracy_threshold and val_loss <= self.loss_threshold:
            print(
                f"\n🎯 Reached target performance! Accuracy: {val_accuracy:.6f}, Loss: {val_loss:.6f}")
            print("✅ Stopping training early due to excellent performance.")
            self.model.stop_training = True


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


def build_model_with_reflection(vocab_size=27, maxlen=50, embed_dim=128):
    """Build a new model from scratch with reflection score integration."""
    print("🔧 Building new model with reflection score integration...")

    # Text input
    text_input = Input(shape=(maxlen,), name='text_input')

    # Reflection score input
    reflection_input = Input(shape=(1,), name='reflection_score_input')

    # Text processing branch
    embedding = Embedding(vocab_size, embed_dim,
                          input_length=maxlen, name='embedding')(text_input)

    # Bidirectional GRU layers
    gru1 = Bidirectional(GRU(128, return_sequences=True),
                         name='bidirectional_1')(embedding)
    dropout1 = Dropout(0.3, name='dropout_1')(gru1)

    gru2 = Bidirectional(GRU(64, return_sequences=False),
                         name='bidirectional_2')(dropout1)
    dropout2 = Dropout(0.3, name='dropout_2')(gru2)

    # Dense layers for text features
    dense1 = Dense(128, activation='relu', name='dense_1')(dropout2)
    dropout3 = Dropout(0.4, name='dropout_3')(dense1)

    dense2 = Dense(64, activation='relu', name='dense_2')(dropout3)
    dropout4 = Dropout(0.3, name='dropout_4')(dense2)

    # Reflection score processing branch
    reflection_expanded = Dense(
        64, activation='relu', name='reflection_expand')(reflection_input)
    reflection_dropout = Dropout(
        0.3, name='reflection_dropout')(reflection_expanded)

    # Combine text and reflection features
    combined = Concatenate(name='combine_features')(
        [dropout4, reflection_dropout])

    # Final processing layers
    final_dense1 = Dense(128, activation='relu',
                         name='final_dense_1')(combined)
    final_dropout1 = Dropout(0.4, name='final_dropout_1')(final_dense1)

    final_dense2 = Dense(64, activation='relu',
                         name='final_dense_2')(final_dropout1)
    final_dropout2 = Dropout(0.3, name='final_dropout_2')(final_dense2)

    # Output layer
    output = Dense(1, activation='sigmoid', name='output')(final_dropout2)

    # Create model
    model = Model(
        inputs=[text_input, reflection_input],
        outputs=output,
        name='model_with_reflection'
    )

    print("✅ Built new model with reflection score integration")
    return model


def generate_training_data_with_reflection(n_samples=20000):
    """Generate comprehensive training data with reflection scores."""
    print(f"🌱 Generating {n_samples} training samples...")

    # Generate balanced dataset
    X_encoded, y_data, symmetry_scores = generate_balanced_multiword_dataset(
        n_samples=n_samples,
        maxlen=50,
        multiword_ratio=0.7
    )

    # The function returns encoded texts and symmetry scores, but we need original texts
    # Let's generate the texts separately
    palindromes = []
    non_palindromes = []

    # Generate palindromes
    while len(palindromes) < n_samples // 2:
        max_words = random.randint(1, 6)
        palindrome = generate_multiword_palindrome(max_words)
        if len(palindrome) <= 50:
            palindromes.append(palindrome)

    # Generate non-palindromes
    while len(non_palindromes) < n_samples // 2:
        max_words = random.randint(1, 6)
        non_palindrome = generate_non_multiword_palindrome(max_words)
        if len(non_palindrome) <= 50:
            non_palindromes.append(non_palindrome)

    # Combine and shuffle
    texts = palindromes + non_palindromes
    labels = [1] * len(palindromes) + [0] * len(non_palindromes)

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    # Convert back to lists for manipulation
    texts = list(texts)
    labels = list(labels)

    # Encode texts
    X_encoded = np.array([encode_multiword(text, maxlen=50) for text in texts])
    y_data = np.array(labels)

    # Calculate reflection scores
    reflection_scores = []
    for text in texts:
        reflection_score = char_reflection_score(text)
        reflection_scores.append(reflection_score)

    reflection_scores = np.array(reflection_scores)

    # Create challenging training data by oversampling cases where reflection score disagrees with actual palindrome status
    challenging_cases = []
    for i, (text, y_true, reflection_score) in enumerate(zip(texts, y_data, reflection_scores)):
        # Check if reflection score is misleading
        is_actual_palindrome = text == text[::-1]
        reflection_threshold = 0.5
        reflection_says_palindrome = reflection_score > reflection_threshold

        # If reflection score disagrees with actual status, this is a challenging case
        if reflection_says_palindrome != is_actual_palindrome:
            challenging_cases.append(i)

    # Oversample challenging cases
    if challenging_cases:
        print(
            f"🎯 Found {len(challenging_cases)} challenging cases to oversample")
        # Add challenging cases multiple times
        for _ in range(5):  # Add each challenging case 5 more times
            for idx in challenging_cases:
                X_encoded = np.vstack([X_encoded, X_encoded[idx:idx+1]])
                y_data = np.append(y_data, y_data[idx])
                reflection_scores = np.append(
                    reflection_scores, reflection_scores[idx])
                texts.append(texts[idx])

    print(f"✅ Generated {len(X_encoded)} total samples")
    return X_encoded, y_data, reflection_scores, texts


def train_model_with_reflection():
    """Train a new model from scratch with reflection score integration."""
    print("🚀 Training new model with reflection score integration...")

    # Build model
    model = build_model_with_reflection()

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )

    # Generate training data
    X_train, y_train, reflection_train, texts_train = generate_training_data_with_reflection(
        25000)

    # Generate validation data
    X_val, y_val, reflection_val, texts_val = generate_training_data_with_reflection(
        5000)

    # Callbacks
    callbacks = [
        PerformanceThresholdCallback(
            accuracy_threshold=0.999, loss_threshold=1e-5),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("🧪 Training model...")
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Validation samples: {len(X_val)}")

    # Train model
    history = model.fit(
        [X_train, reflection_train],
        y_train,
        validation_data=([X_val, reflection_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model_path = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "models", "model_with_reflection_fixed.keras")
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

    # Evaluate on validation set
    val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(
        [X_val, reflection_val], y_val, verbose=0
    )

    print(f"\n📊 Model Evaluation:")
    print(f"📈 Validation Accuracy: {val_acc:.3f}")
    print(f"📈 Validation AUC: {val_auc:.3f}")
    print(f"📈 Validation Loss: {val_loss:.3f}")
    print(f"📈 Validation Precision: {val_precision:.3f}")
    print(f"📈 Validation Recall: {val_recall:.3f}")

    return model, history


def main():
    """Main function for training new model with reflection score."""
    print("🎯 Training New Model with Character Reflection Score Integration")
    print("=" * 70)

    # Train new model
    model, history = train_model_with_reflection()

    if model is not None:
        print("\n🎉 Training completed successfully!")
        print("📁 New model saved with reflection score integration")
    else:
        print("\n❌ Failed to train model")


if __name__ == "__main__":
    main()
