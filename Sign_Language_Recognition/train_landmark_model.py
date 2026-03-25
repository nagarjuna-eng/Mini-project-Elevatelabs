import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# =============================
# CONFIG
# =============================
DATA_DIR = "landmark_data"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "landmark_model.h5")

EPOCHS = 30
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================
# LOAD DATA
# =============================
print("[INFO] Loading landmark data...")

X = []
y = []

for file in sorted(os.listdir(DATA_DIR)):
    if not file.endswith(".csv"):
        continue

    label = file.replace(".csv", "")
    file_path = os.path.join(DATA_DIR, file)

    data = pd.read_csv(file_path, header=None)

    # Each row = 63 features (21 landmarks × x,y,z)
    X.append(data.values)
    y.extend([label] * len(data))

X = np.vstack(X)
y = np.array(y)

print(f"[INFO] Total samples: {len(X)}")
print(f"[INFO] Feature size: {X.shape[1]}")

# =============================
# ENCODE LABELS
# =============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

NUM_CLASSES = y_categorical.shape[1]
print(f"[INFO] Number of classes: {NUM_CLASSES}")

# =============================
# TRAIN / TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# =============================
# BUILD MODEL
# =============================
print("[INFO] Building landmark model...")

model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# TRAIN MODEL
# =============================
print("[INFO] Training model...")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# =============================
# EVALUATE
# =============================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Validation Accuracy: {acc * 100:.2f}%")

# =============================
# SAVE MODEL
# =============================
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)

print(f"[SUCCESS] Landmark model saved at: {MODEL_PATH}")
print("[DONE] Training complete.")