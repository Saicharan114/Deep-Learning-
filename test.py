import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ==========================================
# SETTINGS
# ==========================================

DATASET = "dataset"

IMG_SIZE = 32
NUM_CLASSES = 43

EPOCHS = 10
BATCH_SIZE = 32


# ==========================================
# CREATE MODEL / OUTPUT FOLDERS
# ==========================================

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


print("\n======================================")
print("   TRAFFIC SIGN CNN - TRAINING")
print("======================================\n")


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading training dataset...")

TRAIN_DIR = os.path.join(DATASET, "Train")

images = []
labels = []

print("\nLoading images from class folders...")

total_images = 0

for class_id in range(NUM_CLASSES):

    class_folder = os.path.join(
        TRAIN_DIR,
        str(class_id)
    )

    if not os.path.exists(class_folder):
        print("WARNING: Folder not found:", class_folder)
        continue

    print("Loading class:", class_id)

    for filename in os.listdir(class_folder):

        image_path = os.path.join(
            class_folder,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        images.append(image)
        labels.append(class_id)

        total_images += 1

print("\nTotal training images:", total_images)
print("Number of classes:", NUM_CLASSES)


# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(
    images,
    dtype=np.float32
)

y = np.array(labels)

print("\nImage shape:", X.shape)
print("Label shape:", y.shape)


# ==========================================
# NORMALIZATION
# ==========================================

X = X / 255.0


# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# ONE-HOT ENCODING
# ==========================================

y_train = to_categorical(
    y_train,
    NUM_CLASSES
)

y_val = to_categorical(
    y_val,
    NUM_CLASSES
)


print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ==========================================
# CNN MODEL
# ==========================================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(32, 32, 3)
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# MODEL SUMMARY
# ==========================================

print("\nCNN Model:")
model.summary()


# ==========================================
# TRAIN
# ==========================================

print("\n======================================")
print("         STARTING TRAINING")
print("======================================\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# ==========================================
# SAVE MODEL
# ==========================================

model.save(
    "model/traffic_sign_model.keras"
)


# ==========================================
# FINAL TRAINING RESULTS
# ==========================================

train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

train_loss = history.history["loss"][-1]
val_loss = history.history["val_loss"][-1]


print("\n======================================")
print("          TRAINING COMPLETED")
print("======================================")

print(
    "Final Training Accuracy: {:.2f}%".format(
        train_accuracy * 100
    )
)

print(
    "Final Validation Accuracy: {:.2f}%".format(
        val_accuracy * 100
    )
)

print(
    "Final Training Loss: {:.4f}".format(
        train_loss
    )
)

print(
    "Final Validation Loss: {:.4f}".format(
        val_loss
    )
)

print("\nModel saved at:")
print("model/traffic_sign_model.keras")


# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.legend()
plt.grid()

plt.savefig(
    "outputs/training_accuracy.png",
    dpi=200
)

plt.show()


# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "CNN Training and Validation Loss"
)

plt.legend()
plt.grid()

plt.savefig(
    "outputs/training_loss.png",
    dpi=200
)

plt.show()


print("\nTraining graphs saved in outputs/")
