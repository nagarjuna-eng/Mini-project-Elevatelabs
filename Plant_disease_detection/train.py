import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import json

# ------------------------
# Hyperparameters
# ------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12

# ------------------------
# Data Generator
# ------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ------------------------
# Save class indices
# ------------------------
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("Class Indices:", train_data.class_indices)

# ------------------------
# CNN Model
# ------------------------
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),

    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),

    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------
# Train Model
# ------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ------------------------
# Save Model (NEW FORMAT)
# ------------------------
model.save("plant_disease_model.keras")

print("✅ Model and class indices saved successfully")