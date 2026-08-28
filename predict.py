import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# ==========================================
# SETTINGS
# ==========================================

DATASET = "dataset"

TEST_DIR = os.path.join(
    DATASET,
    "Test"
)

MODEL_PATH = "model/traffic_sign_model.keras"

IMG_SIZE = 32

NUMBER_OF_IMAGES = 9


# ==========================================
# TRAFFIC SIGN NAMES
# ==========================================

classes = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "Speed Limit 60",
    4: "Speed Limit 70",
    5: "Speed Limit 80",
    6: "End Speed Limit 80",
    7: "Speed Limit 100",
    8: "Speed Limit 120",
    9: "No Passing",
    10: "No Passing >3.5t",
    11: "Right of Way",
    12: "Priority Road",
    13: "Yield",
    14: "STOP",
    15: "No Vehicles",
    16: "Vehicles >3.5t",
    17: "No Entry",
    18: "General Caution",
    19: "Curve Left",
    20: "Curve Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles",
    30: "Ice/Snow",
    31: "Wild Animals",
    32: "End Speed Limits",
    33: "Turn Right",
    34: "Turn Left",
    35: "Ahead Only",
    36: "Straight or Right",
    37: "Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout",
    41: "End No Passing",
    42: "End No Passing >3.5t"
}


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ==========================================
# CHECK MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model file not found:")
    print(MODEL_PATH)

    exit()


# ==========================================
# CHECK TEST FOLDER
# ==========================================

if not os.path.exists(TEST_DIR):

    print("ERROR: Test folder not found:")
    print(TEST_DIR)

    exit()


# ==========================================
# LOAD MODEL
# ==========================================

print("\n======================================")
print("   TRAFFIC SIGN RECOGNITION")
print("======================================\n")

print("Loading trained model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# GET TEST IMAGES
# ==========================================

print("\nReading test images...")

image_files = []

for filename in os.listdir(TEST_DIR):

    file_path = os.path.join(
        TEST_DIR,
        filename
    )

    if os.path.isfile(file_path):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in [
            ".png",
            ".jpg",
            ".jpeg",
            ".ppm",
            ".bmp"
        ]:

            image_files.append(
                filename
            )


# Sort images

image_files.sort()


print(
    "Total test images found:",
    len(image_files)
)


# ==========================================
# CHECK NUMBER OF IMAGES
# ==========================================

if len(image_files) == 0:

    print("\nERROR: No test images found.")

    exit()


# ==========================================
# SELECT 9 IMAGES
# ==========================================

number_to_select = min(
    NUMBER_OF_IMAGES,
    len(image_files)
)


selected_images = image_files[
    :number_to_select
]


print(
    "\nSelected images:",
    number_to_select
)


# ==========================================
# CREATE OUTPUT FIGURE
# ==========================================

rows = 3
cols = 3

plt.figure(
    figsize=(14, 12)
)


# ==========================================
# PREDICT IMAGES
# ==========================================

for i, filename in enumerate(
    selected_images
):

    image_path = os.path.join(
        TEST_DIR,
        filename
    )


    # ======================================
    # LOAD IMAGE
    # ======================================

    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            "\nCould not load:",
            filename
        )

        continue


    # ======================================
    # CONVERT BGR TO RGB
    # ======================================

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # ======================================
    # RESIZE
    # ======================================

    resized = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE)
    )


    # ======================================
    # NORMALIZE
    # ======================================

    normalized = (
        resized.astype(
            np.float32
        ) / 255.0
    )


    # ======================================
    # ADD BATCH DIMENSION
    # ======================================

    input_image = np.expand_dims(
        normalized,
        axis=0
    )


    # ======================================
    # PREDICTION
    # ======================================

    prediction = model.predict(
        input_image,
        verbose=0
    )


    # ======================================
    # GET CLASS
    # ======================================

    predicted_class = int(
        np.argmax(
            prediction,
            axis=1
        )[0]
    )


    # ======================================
    # GET CONFIDENCE
    # ======================================

    confidence = float(
        np.max(
            prediction
        ) * 100
    )


    predicted_name = classes.get(
        predicted_class,
        "Unknown"
    )


    # ======================================
    # TERMINAL OUTPUT
    # ======================================

    print("\n------------------------------")

    print(
        "Image:",
        filename
    )

    print(
        "Predicted class:",
        predicted_class
    )

    print(
        "Predicted sign:",
        predicted_name
    )

    print(
        "Confidence: {:.2f}%".format(
            confidence
        )
    )


    # ======================================
    # DISPLAY IMAGE
    # ======================================

    plt.subplot(
        rows,
        cols,
        i + 1
    )

    plt.imshow(
        image_rgb
    )

    plt.title(
        "Predicted:\n{}\nConfidence: {:.1f}%".format(
            predicted_name,
            confidence
        ),
        fontsize=9
    )

    plt.axis("off")


# ==========================================
# TITLE
# ==========================================

plt.suptitle(
    "Traffic Sign Recognition Using CNN",
    fontsize=18
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.95
    ]
)


# ==========================================
# SAVE RESULT
# ==========================================

output_path = (
    "outputs/"
    "9_traffic_sign_predictions.png"
)


plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


# ==========================================
# SHOW RESULT
# ==========================================

plt.show()


# ==========================================
# COMPLETED
# ==========================================

print("\n======================================")
print("       PREDICTION COMPLETED")
print("======================================")

print(
    "Images displayed:",
    number_to_select
)

print("\nResult saved at:")

print(
    output_path
)

print("======================================")
