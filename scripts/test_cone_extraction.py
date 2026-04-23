"""
visualize_processed_dataset.py

This script visualizes samples from the processed cone dataset.

It performs the following steps:
    1. Loads the processed annotations CSV
    2. Randomly selects a sample
    3. Loads the cropped image
    4. Prints image information
    5. Draws bounding boxes on the image
    6. Displays the result

This script is meant for debugging and validating the preprocessing pipeline.
"""

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from pathlib import Path        # modern path handling
import pandas as pd             # reading annotation CSV
import numpy as np              # numerical arrays
import matplotlib.pyplot as plt # visualization
import random                   # selecting random samples
import cv2                      # drawing bounding boxes


# ---------------------------------------------------------
# Define dataset paths
# ---------------------------------------------------------

# Root directory of the processed dataset
dataset_root = Path(
    r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset"
)

# Directory containing cropped images
images_dir = dataset_root / "images"

# Optional mask directory (only if you saved masks)
masks_dir = dataset_root / "masks"

# CSV with updated bounding boxes
annotations_csv = dataset_root / "annotations_cropped_1603_before_leaving.csv"


# ---------------------------------------------------------
# Load annotations CSV
# ---------------------------------------------------------

df = pd.read_csv(annotations_csv)

print("Number of samples in dataset:", len(df))


# ---------------------------------------------------------
# Select a random sample
# ---------------------------------------------------------

row = df.sample(1).iloc[0]

# Filename of processed image
image_filename = row["processed_image"]

# Construct full image path
image_path = images_dir / image_filename

print("\nSelected sample:")
print("Image file:", image_filename)


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

img = np.load(image_path)

print("\nImage information:")
print("-------------------")
print("Shape:", img.shape)
print("Datatype:", img.dtype)
print("Min pixel value:", img.min())
print("Max pixel value:", img.max())


# ---------------------------------------------------------
# Extract bounding boxes
# ---------------------------------------------------------

bbox1 = [
    int(row["box1_x1"]),
    int(row["box1_y1"]),
    int(row["box1_x2"]),
    int(row["box1_y2"])
]

bbox2 = [
    int(row["box2_x1"]),
    int(row["box2_y1"]),
    int(row["box2_x2"]),
    int(row["box2_y2"])
]

CLASS_NAMES = {
    1: "A2C_hinge_left",
    2: "A2C_hinge_right",
    3: "A3C_hinge_bottom",
    4: "A3C_hinge_top",
    5: "A4C_hinge_left",
    6: "A4C_hinge_right",
    7: "PSAX_MV_mitral_valve",
    8: "PLAX_hinge_bottom",
    9: "PLAX_hinge_top"
}

class1 = int(row['label1'])
class2 = int(row['label2'])

label1_name = CLASS_NAMES.get(class1, "unknown")
label2_name = CLASS_NAMES.get(class2, "unknown")


# ---------------------------------------------------------
# Prepare image for drawing
# ---------------------------------------------------------

# Copy image to avoid modifying original
vis_img = img.copy()

# If image is float convert to uint8 for OpenCV drawing
if vis_img.dtype != np.uint8:
    vis_img = vis_img.astype(np.uint8)


# ---------------------------------------------------------
# Draw bounding boxes
# ---------------------------------------------------------

# Draw first bounding box (green)
cv2.rectangle(
    vis_img,
    (bbox1[0], bbox1[1]),
    (bbox1[2], bbox1[3]),
    (0, 255, 0),
    2
)

# Draw bbox1 label
cv2.putText(
    vis_img,
    label1_name,
    (bbox1[0], bbox1[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0,255,0),
    2
)

# Draw second bounding box (red)
cv2.rectangle(
    vis_img,
    (bbox2[0], bbox2[1]),
    (bbox2[2], bbox2[3]),
    (255, 0, 0),
    2
)

cv2.putText(
    vis_img,
    label2_name,
    (bbox2[0], bbox2[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255,0,0),
    2
)

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
print(f"Processed image: {image_filename}")

plt.figure(figsize=(8, 8))

plt.imshow(vis_img)
plt.title("Processed Image with Bounding Boxes")

#plt.axis("off")

plt.show()
