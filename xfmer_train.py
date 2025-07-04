from data_gen import generate_balanced_dataset, preprocess
from palindrome_transformer import build_hybrid_model
import tensorflow as tf
from sklearn.utils import class_weight
import numpy as np
import os

# Model config
MAXLEN = 12
model = build_hybrid_model(input_len=MAXLEN, vocab_size=27)
model.compile(optimizer="adam", loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=["accuracy"])

# Curriculum stages
curriculum = [
    ("easy",   2000,  500, 4),   # 85% positive, trivial
    ("medium", 5000,  750, 6),   # 70% positive, light negatives
    ("hard1",  9000, 1000, 8),   # 60% positive, moderate difficulty
    ("hard2",  12000, 1000, 10),  # 50/50 mix, no boosting
]
for stage, n_train, n_val, n_epochs in curriculum:
    print(f"\n📚 Curriculum Stage: '{stage}' ({n_epochs} epochs)")
    
    train_data = generate_balanced_dataset(n_samples=n_train, maxlen=MAXLEN, stage=stage)
    val_data = generate_balanced_dataset(n_samples=n_val, maxlen=MAXLEN, stage=stage)
    
    X_train, y_train = preprocess(train_data, maxlen=MAXLEN)
    X_val, y_val = preprocess(val_data, maxlen=MAXLEN)
    
    model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=n_epochs,
        validation_data=(X_val, y_val)
    )

print("\n🎯 Final fine-tuning on full dataset")
final_train = generate_balanced_dataset(n_samples=70000, maxlen=MAXLEN, stage="full")
final_val = generate_balanced_dataset(n_samples=1000, maxlen=MAXLEN, stage="full")

X_final_train, y_final_train = preprocess(final_train, maxlen=MAXLEN)
X_final_val, y_final_val = preprocess(final_val, maxlen=MAXLEN)

#  Freeze early transformer layers
for layer in model.layers[:3]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=1.0),
    metrics=["accuracy"]
)


y_np = y_final_train.flatten()
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_np),
    y=y_np
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(
    X_final_train, y_final_train,
    batch_size=64,
    epochs=20,
    validation_data=(X_final_val, y_final_val),
    callbacks=[early_stop],
    class_weight=class_weights_dict
)

print("\n🔍 Misclassifications in validation set with thresholding:")
preds = model.predict(X_final_val)

index_to_char = {i+1: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
decoded = ["".join(index_to_char.get(idx, "_") for idx in seq) for seq in X_final_val]

for i, pred in enumerate(preds):
    conf = pred[0]
    truth = y_final_val[i]
    word = decoded[i]

    if conf >= 0.7:
        decision = 1
    elif conf <= 0.3:
        decision = 0
    else:
        decision = 'unsure'

    if decision != truth:
        print(f"❌ '{word}' → pred={decision}, truth={truth}, conf={conf:.2f}")

model.save("polished_model.keras")

