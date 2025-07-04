import os
import random
import string
import time
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from data_gen import generate_string, preprocess
from model import LearnedPositionalEncoding
import json
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
MAXLEN = int(os.getenv("MAXLEN", 12))
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
MODEL_PATH = "polished_model.keras"

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

def append_unique(filename, entries):
    if not entries:
        return 0
    try:
        with open(filename, "r") as f:
            existing = set(line.strip() for line in f)
    except FileNotFoundError:
        existing = set()

    new_entries = entries - existing
    if new_entries:
        with open(filename, "a") as f:
            for entry in sorted(new_entries):
                f.write(entry + "\n")
    return len(new_entries)

def retrain_model_with_hard_examples(model):
    print("\n🔁 Re-training with curriculum strategy...")

    # Load base data
    palindromes = [w for w in load_words("palindromes.txt") if len(w) <= MAXLEN]
    not_palindromes = [w for w in load_words("not_palindromes.txt") if len(w) <= MAXLEN]
    full_data = [(w, 1) for w in palindromes] + [(w, 0) for w in not_palindromes]

    # Load hard examples
    hard_data = []
    if os.path.exists("hard_negatives.txt"):
        for line in open("hard_negatives.txt"):
            word = line.strip().split()[0]
            if word and len(word) <= MAXLEN:
                hard_data.append((word, 0))
    if os.path.exists("hard_positives.txt"):
        for line in open("hard_positives.txt"):
            word = line.strip().split()[0]
            if word and len(word) <= MAXLEN:
                hard_data.append((word, 1))

    # Phase 1: Train on hard examples only
    print(f"📘 Phase 1: Hard example warm-up ({len(hard_data)} samples)")
    X_hard, y_hard = preprocess(hard_data, maxlen=MAXLEN)
    val_data = generate_test_set(count=500, maxlen=MAXLEN)
    X_val, y_val = preprocess(val_data, maxlen=MAXLEN)

    y_np = np.array(y_hard).flatten()
    class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_np), y=y_np)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=BinaryFocalCrossentropy(gamma=1.0),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    model.fit(
        X_hard, y_hard,
        batch_size=32,
        epochs=5,
        validation_data=(X_val, y_val),
        class_weight=class_weights_dict,
        verbose=2
    )

    # Phase 2: Train on blended full set
    blended_data = full_data + hard_data
    print(f"📘 Phase 2: Full dataset + hard examples ({len(blended_data)} samples)")
    random.shuffle(blended_data)
    X_train, y_train = preprocess(blended_data, maxlen=MAXLEN)

    y_np = np.array(y_train).flatten()
    class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_np), y=y_np)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=15,  # continue from previous 5, total = 20
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        class_weight=class_weights_dict,
        verbose=2
    )

    model.save("polished_retrained_model.keras")
    print("\n✅ Model retrained and saved.")


def predict_and_print(model, test_set, threshold=THRESHOLD):
    print(f"{'Word':<15} {'Expected':<18} {'Prediction':<22} {'Confidence':<12} {'Time (s)'}")
    print("-" * 80)

    correct = 0
    y_true = []
    y_pred = []
    new_hard_negatives = set()
    new_hard_positives = set()

    for word, label in test_set:
        start = time.time()
        confidence = model.predict(encode_word(word), verbose=0)[0][0]
        end = time.time()

        pred = int(confidence > threshold)
        expected = ["Not a palindrome", "Palindrome"][label]
        predicted = ["❌ Not a palindrome", "✅ Palindrome"][pred]

        y_true.append(label)
        y_pred.append(pred)

        if pred == label:
            correct += 1
        else:
            print(f"{word:<15} {expected:<18} {predicted:<22} ({confidence:.2f})     {end - start:.3f}s ❌")

            if label == 0 and pred == 1 and confidence > threshold:
                new_hard_negatives.add(f"{word} ({confidence:.2f})")
            elif label == 1 and pred == 0 and confidence < (1 - threshold):
                new_hard_positives.add(f"{word} ({confidence:.2f})")

    # Save hard negatives
    if new_hard_negatives:
        existing = set()
        if os.path.exists("hard_negatives.txt"):
            with open("hard_negatives.txt", "r") as f:
                existing = set(line.strip() for line in f)
        all_hard = sorted(existing.union(new_hard_negatives))
        with open("hard_negatives.txt", "w") as f:
            f.write("\n".join(all_hard) + "\n")
        print(f"\n💾 Logged {len(new_hard_negatives)} new hard negatives (total now {len(all_hard)}).")

    # Save hard positives
    if new_hard_positives:
        existing = set()
        if os.path.exists("hard_positives.txt"):
            with open("hard_positives.txt", "r") as f:
                existing = set(line.strip() for line in f)
        all_hard = sorted(existing.union(new_hard_positives))
        with open("hard_positives.txt", "w") as f:
            f.write("\n".join(all_hard) + "\n")
        print(f"\n💾 Logged {len(new_hard_positives)} new hard positives (total now {len(all_hard)}).")

    # Trigger retrain if needed
    if new_hard_negatives or new_hard_positives:
        retrain_model_with_hard_examples(model)

    # Metrics and logging
    acc = correct / len(test_set)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n🎯 Accuracy: {correct}/{len(test_set)} = {acc:.2%}")
    print("📊 Confusion Matrix:\n", cm)

    # Log to JSON
    log_file = "stress_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log = json.load(f)
    else:
        log = {"rounds": []}

    log["rounds"].append({
        "round": len(log["rounds"]) + 1,
        "accuracy": round(acc, 4),
        "FP": int(fp),
        "FN": int(fn)
    })

    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "LearnedPositionalEncoding": LearnedPositionalEncoding,
            "BinaryFocalCrossentropy": BinaryFocalCrossentropy,
        }
    )

    for i in range(10):
        print(f"\n🔁 Round {i+1}/10")
        palindromes = [w for w in load_words("palindromes.txt") if len(w) <= MAXLEN]
        not_palindromes = [w for w in load_words("not_palindromes.txt") if len(w) <= MAXLEN]

        sample = [(w, 1) for w in random.sample(palindromes, 200)]
        sample += [(w, 0) for w in random.sample(not_palindromes, 200)]
        generated = generate_test_set(count=200)
        full_test = sample + generated
        random.shuffle(full_test)

        predict_and_print(model, full_test)

    # Optional visualization here, or separate script
