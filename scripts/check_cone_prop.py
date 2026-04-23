import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION (EDIT THESE PATHS)
# --------------------------------------------------

images_dir = Path(r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\images")
csv_path = Path(r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\annotations_propagated_aSaco.csv")

NUM_SAMPLES = 30  # how many samples to visualize


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(csv_path)

print(f"[INFO] Total samples in dataset: {len(df)}")

# Sample subset
df_sample = df.sample(NUM_SAMPLES, random_state=42).reset_index(drop=True)


# --------------------------------------------------
# DRAW FUNCTION
# --------------------------------------------------

def draw_bboxes(image, bboxes, labels):
    """
    Draw bounding boxes + labels on image
    """

    img = image.copy()

    for i, (box, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = map(int, box)

        # Different colors per bbox
        color = (0, 255, 0) if i == 0 else (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            img,
            f"L{label}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return img


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

for i, row in df_sample.iterrows():

    img_path = images_dir / row["processed_image"]

    if not img_path.exists():
        print(f"[WARNING] Missing image: {img_path}")
        continue

    # Load image
    image = np.load(img_path)

    # Ensure 3 channels for visualization
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)

    # Normalize if needed
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Extract bounding boxes
    bboxes = [
        [row["box1_x1"], row["box1_y1"], row["box1_x2"], row["box1_y2"]],
        [row["box2_x1"], row["box2_y1"], row["box2_x2"], row["box2_y2"]],
    ]

    labels = [row["label1"], row["label2"]]

    # Draw boxes
    vis_img = draw_bboxes(image, bboxes, labels)

    # Resize for display
    vis_img = cv2.resize(vis_img, (512, 512))

    # --------------------------------------------------
    # TITLE / INFO (VERY IMPORTANT)
    # --------------------------------------------------

    title = (
        f"[{i+1}/{len(df_sample)}] "
        f"study={row['study_id']} | view={row['view']} | "
        f"frame={row['frame_index']} | offset={row['offset']} | "
        f"labels=({row['label1']},{row['label2']})"
    )

    print("\n" + "="*80)
    print(title)
    print(f"Image: {row['processed_image']}")
    print(f"BBoxes: {bboxes}")

    # Show image
    cv2.imshow("Dataset Debug Viewer", vis_img)
    cv2.setWindowTitle("Dataset Debug Viewer", title)

    key = cv2.waitKey(0)

    # Controls
    if key == 27:  # ESC
        print("[EXIT] User terminated visualization.")
        break
    elif key == ord('q'):
        print("[EXIT] User terminated visualization.")
        break
    elif key == ord('s'):
        # Save current debug image
        save_path = images_dir / f"DEBUG_{row['processed_image']}.png"
        cv2.imwrite(str(save_path), vis_img)
        print(f"[SAVED] {save_path}")

cv2.destroyAllWindows()