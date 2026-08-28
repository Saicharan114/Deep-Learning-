import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


# ==========================================
# SETTINGS
# ==========================================

DATASET = "dataset"

TRAIN_DIR = os.path.join(
    DATASET,
    "Train"
)

MODEL_PATH = "model/traffic_sign_model.keras"

IMG_SIZE = 32

NUM_CLASSES = 43

VALIDATION_SIZE = 0.2


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ==========================================
# START
# ==========================================

print("\n======================================")
print("     TRAFFIC SIGN MODEL EVALUATION")
print("======================================\n")


# ==========================================
# CHECK MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found:")
    print(MODEL_PATH)

    exit()


# ==========================================
# CHECK TRAIN FOLDER
# ==========================================

if not os.path.exists(TRAIN_DIR):

    print("ERROR: Train folder not found:")
    print(TRAIN_DIR)

    exit()


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# LOAD LABELED TRAIN DATA
# ==========================================

print("\nLoading images from Train folders...")

images = []
labels = []

total_images = 0


for class_id in range(NUM_CLASSES):

    class_folder = os.path.join(
        TRAIN_DIR,
        str(class_id)
    )

    if not os.path.exists(class_folder):

        print(
            "WARNING: Folder not found:",
            class_folder
        )

        continue


    print(
        "Loading class:",
        class_id
    )


    for filename in os.listdir(
        class_folder
    ):

        image_path = os.path.join(
            class_folder,
            filename
        )


        if not os.path.isfile(
            image_path
        ):
            continue


        extension = os.path.splitext(
            filename
        )[1].lower()


        if extension not in [
            ".png",
            ".jpg",
            ".jpeg",
            ".ppm",
            ".bmp"
        ]:
            continue


        image = cv2.imread(
            image_path
        )


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


        images.append(
            image
        )


        labels.append(
            class_id
        )


        total_images += 1


print("\nTotal images loaded:")
print(total_images)


# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(
    images,
    dtype=np.float32
)

y = np.array(
    labels,
    dtype=np.int32
)


print(
    "\nComplete dataset shape:",
    X.shape
)

print(
    "Complete labels shape:",
    y.shape
)


# ==========================================
# NORMALIZE
# ==========================================

X = X / 255.0


# ==========================================
# CREATE VALIDATION DATA
# ==========================================

print("\nCreating validation dataset...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=42,
    stratify=y
)


print(
    "Training images:",
    len(X_train)
)

print(
    "Validation images:",
    len(X_val)
)
# ==========================================
# ONE-HOT ENCODE VALIDATION LABELS
# ==========================================
y_val_original = y_val.copy()
y_val = to_categorical(
    y_val,
    NUM_CLASSES
)

# ==========================================
# EVALUATE MODEL
# ==========================================

print("\n======================================")
print("       EVALUATING MODEL")
print("======================================\n")


loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=1
)


# ==========================================
# RESULTS
# ==========================================

print("\n======================================")
print("          EVALUATION RESULTS")
print("======================================")

print(
    "Validation Accuracy: {:.2f}%".format(
        accuracy * 100
    )
)

print(
    "Validation Loss: {:.4f}".format(
        loss
    )
)


# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_val,
    verbose=1
)


predicted_classes = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# CORRECT / WRONG
# ==========================================

correct = np.sum(
    predicted_classes == y_val_original
)

wrong = np.sum(
    predicted_classes != y_val_original
)


print("\n======================================")
print("       PREDICTION RESULTS")
print("======================================")

print(
    "Correct predictions:",
    correct
)

print(
    "Wrong predictions:",
    wrong
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_val_original,
    predicted_classes,
    labels=np.arange(NUM_CLASSES)
)


# ==========================================
# DISPLAY CONFUSION MATRIX
# ==========================================

plt.figure(
    figsize=(14, 12)
)

plt.imshow(
    cm
)

plt.title(
    "Traffic Sign Classification Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.colorbar()

plt.xticks(
    np.arange(NUM_CLASSES)
)

plt.yticks(
    np.arange(NUM_CLASSES)
)

plt.tight_layout()


# ==========================================
# SAVE CONFUSION MATRIX
# ==========================================

output_path = (
    "outputs/"
    "confusion_matrix.png"
)


plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()


# ==========================================
# FINAL
# ==========================================

print("\n======================================")
print("      EVALUATION COMPLETED")
print("======================================")

print(
    "Validation Accuracy: {:.2f}%".format(
        accuracy * 100
    )
)

print(
    "Correct predictions:",
    correct
)

print(
    "Wrong predictions:",
    wrong
)

print("\nConfusion matrix saved at:")

print(
    output_path
)

print("======================================")
