from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm

from src.models.faster_rcnn_model import build_faster_rcnn_model


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

frames_dir = Path(rf"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids\frames_PLAX")   # <-- CHANGE
output_video_path = Path(rf"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids\inferences\inference_PLAX_best_model.mp4")

checkpoint_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments_after_vacations\3classes_3warmup_15constant_cosine_200426_full\checkpoints\best_match_precision.pth")  # <-- CHANGE

score_threshold = 0.5
target_size = (224, 224)  # (H, W)
fps = 10  # fallback 


# --------------------------------------------------
# CLASS MAP + COLORS
# --------------------------------------------------

CLASS_MAP = {
    1: "HINGE",
    2: "PSAX_MV",
    0: "background"
}

CLASS_COLORS = {
    1: (0, 0, 255),     # red
    2: (255, 0, 0),     # blue
    0: (128, 128, 128)
}


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_faster_rcnn_model(num_classes=3)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

print("[INFO] Model loaded")


# --------------------------------------------------
# GET FRAME LIST
# --------------------------------------------------

frame_paths = sorted(frames_dir.glob("*.npy"))

if len(frame_paths) == 0:
    raise RuntimeError("No .npy frames found")

print(f"[INFO] Found {len(frame_paths)} frames")


# --------------------------------------------------
# VIDEO WRITER
# --------------------------------------------------



panel_width = 250

video_writer = cv2.VideoWriter(
    str(output_video_path),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (target_size[1] + panel_width, target_size[0])  # <-- FIX
)

# --------------------------------------------------
# DRAW FUNCTION (OpenCV version)
# --------------------------------------------------

def get_color_from_index(i):
    colors = [
        (255, 0, 0),    # blue
        (0, 255, 0),    # green
        (0, 0, 255),    # red
        (255, 255, 0),  # cyan
        (255, 0, 255),  # magenta
        (0, 255, 255),  # yellow
    ]
    return colors[i % len(colors)]

def draw_prediction_panel(img, boxes, labels, scores, image_width):
    """
    Draw prediction panel on the RIGHT side (outside image).
    """

    h = img.shape[0]
    panel_width = img.shape[1] - image_width

    panel_x1 = image_width + 10
    panel_x2 = image_width + panel_width - 10

    y1 = 10

    # Background (solid, cleaner than transparent)
    cv2.rectangle(
        img,
        (image_width, 0),
        (img.shape[1], h),
        (0, 0, 0),
        -1
    )

    # Title
    cv2.putText(
        img,
        "Predictions",
        (panel_x1, y1 + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Draw each prediction
    for i in range(len(boxes)):

        score = float(scores[i])
        label = int(labels[i])
        class_name = CLASS_MAP.get(label, "UNK")
        color = get_color_from_index(i)

        text = f"[{i}] {class_name} {score:.2f}"

        cv2.putText(
            img,
            text,
            (panel_x1, y1 + 40 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    # Total count
    cv2.putText(
        img,
        f"Total: {len(boxes)}",
        (panel_x1, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    return img

def draw_predictions(image, boxes, labels, scores):

    img = image.copy()

    for i in range(len(boxes)):

        x1, y1, x2, y2 = map(int, boxes[i])
        label = int(labels[i])
        score = float(scores[i])

        color = get_color_from_index(i)
        class_name = CLASS_MAP.get(label, "UNK")

        # BOX
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # TEXT
        text = f"[{i}] {score:.2f}"

        cv2.putText(
            img,
            text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return img


# --------------------------------------------------
# MAIN INFERENCE LOOP
# --------------------------------------------------

print("[INFO] Running inference...")

for frame_path in tqdm(frame_paths):

    # -----------------------------
    # LOAD FRAME
    # -----------------------------
    img = np.load(frame_path)

    # Ensure shape (H, W, C)
    if img.ndim == 2:
        img = img[..., None]

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2) 

    # Normalize if needed (depends on training)
    img_float = img.astype(np.float32) / 255.0

    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1)

    tensor = tensor.to(device)

    # -----------------------------
    # INFERENCE
    # -----------------------------
    with torch.no_grad():
        predictions = model([tensor])

    pred = predictions[0]
    

    boxes = pred["boxes"].cpu()
    labels = pred["labels"].cpu()
    scores = pred["scores"].cpu()

    # -----------------------------
    # FILTER BY SCORE
    # -----------------------------
    keep = scores > score_threshold

    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if len(scores) > 0:
        sorted_idx = scores.argsort(descending=True)

        boxes = boxes[sorted_idx]
        labels = labels[sorted_idx]
        scores = scores[sorted_idx]

    # -----------------------------
    # DRAW
    # -----------------------------
    vis = img.copy()

    # Normalize for visualization
    if vis.dtype != np.uint8:
        vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
        vis = vis.astype(np.uint8)

    # Draw boxes first
    vis = draw_predictions(vis, boxes, labels, scores)

    h, w = vis.shape[:2]
    panel_width = 250

    # Create extended canvas
    canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    canvas[:, :w] = vis

    # Draw panel on the right
    canvas = draw_prediction_panel(canvas, boxes, labels, scores, image_width=w)

    # -----------------------------
    # WRITE FRAME
    # -----------------------------
    video_writer.write(canvas)


# --------------------------------------------------
# FINALIZE
# --------------------------------------------------

video_writer.release()

print("\n--------------------------------------")
print("INFERENCE COMPLETE")
print("--------------------------------------")
print(f"Frames processed: {len(frame_paths)}")
print(f"Saved video: {output_video_path}")
print("--------------------------------------")