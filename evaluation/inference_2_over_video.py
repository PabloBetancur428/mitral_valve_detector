from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm

from src.models.faster_rcnn_model import build_faster_rcnn_model


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

frames_dir = Path(rf"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids\frames_psax_mv")   # <-- CHANGE
output_video_path = Path(rf"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids\inferences\PSAX_MV_new_rule_track.mp4")

checkpoint_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments_after_vacations\3classes_3warmup_15constant_cosine_200426_full\checkpoints\best_match_precision.pth")  # <-- CHANGE
#checkpoint_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments_after_vacations\sgd_lr0.01_bs4_cosine_3classes_3warmup_7constant_cosine_160426_nonstop\checkpoints\best_match_precision.pth")  # <-- CHANGE

score_threshold = 0.5
target_size = (224, 224)  # (H, W)
fps = 10  # fallback 
dominant_label_threshold = 0.75
track_iou_threshold = 0.30


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

def get_color_from_track_id(track_id):
    colors = [
        (255, 0, 0),    # blue
        (0, 255, 0),    # green
        (0, 0, 255),    # red
        (255, 255, 0),  # cyan
        (255, 0, 255),  # magenta
        (0, 255, 255),  # yellow
    ]
    return colors[track_id % len(colors)]


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def prepare_boxes_for_rendering(frame_result, locked_label):
    boxes = frame_result["boxes"]
    labels = frame_result["labels"]
    scores = frame_result["scores"]

    if locked_label is not None:
        keep = labels == locked_label
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

    return boxes, labels, scores


def assign_track_ids(frame_results, locked_label, iou_threshold):
    next_track_id = 0
    prev_boxes = np.empty((0, 4), dtype=np.float32)
    prev_track_ids = np.empty((0,), dtype=np.int32)

    for frame_result in frame_results:
        boxes, labels, scores = prepare_boxes_for_rendering(frame_result, locked_label)
        track_ids = np.full(len(boxes), -1, dtype=np.int32)

        candidate_matches = []
        if len(prev_boxes) > 0 and len(boxes) > 0:
            for prev_idx, prev_box in enumerate(prev_boxes):
                best_iou = 0.0
                best_curr_idx = -1

                for curr_idx, curr_box in enumerate(boxes):
                    iou = compute_iou(prev_box, curr_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_curr_idx = curr_idx

                if best_curr_idx != -1 and best_iou > iou_threshold:
                    candidate_matches.append((best_iou, prev_idx, best_curr_idx))

        candidate_matches.sort(reverse=True)
        used_curr_idxs = set()

        for _, prev_idx, curr_idx in candidate_matches:
            if curr_idx in used_curr_idxs:
                continue

            track_ids[curr_idx] = prev_track_ids[prev_idx]
            used_curr_idxs.add(curr_idx)

        for curr_idx in range(len(boxes)):
            if track_ids[curr_idx] == -1:
                track_ids[curr_idx] = next_track_id
                next_track_id += 1

        frame_result["render_boxes"] = boxes
        frame_result["render_labels"] = labels
        frame_result["render_scores"] = scores
        frame_result["track_ids"] = track_ids

        prev_boxes = boxes
        prev_track_ids = track_ids


def draw_prediction_panel(img, boxes, labels, scores, track_ids, image_width):
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
        track_id = int(track_ids[i])
        class_name = CLASS_MAP.get(label, "UNK")
        color = get_color_from_track_id(track_id)

        text = f"[T{track_id}] {class_name} {score:.2f}"

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

def draw_predictions(image, boxes, labels, scores, track_ids):

    img = image.copy()

    for i in range(len(boxes)):

        x1, y1, x2, y2 = map(int, boxes[i])
        label = int(labels[i])
        score = float(scores[i])
        track_id = int(track_ids[i])

        color = get_color_from_track_id(track_id)
        class_name = CLASS_MAP.get(label, "UNK")

        # BOX
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # TEXT
        text = f"[T{track_id}] {score:.2f}"

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


def load_frame(frame_path):
    img = np.load(frame_path)

    # Ensure shape (H, W, C)
    if img.ndim == 2:
        img = img[..., None]

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    return img


def run_inference_on_frame(img):
    img_float = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).to(device)

    with torch.no_grad():
        predictions = model([tensor])

    pred = predictions[0]

    boxes = pred["boxes"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()

    keep = scores > score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if len(scores) > 0:
        sorted_idx = np.argsort(-scores)
        boxes = boxes[sorted_idx]
        labels = labels[sorted_idx]
        scores = scores[sorted_idx]

    return boxes, labels, scores


# --------------------------------------------------
# FIRST PASS: RUN INFERENCE + DECIDE VIDEO LABEL
# --------------------------------------------------

print("[INFO] Running inference pass...")

frame_results = []
top_label_counts = {
    1: 0,  # HINGE
    2: 0,  # PSAX_MV
}

for frame_path in tqdm(frame_paths, desc="Inference pass"):
    img = load_frame(frame_path)
    boxes, labels, scores = run_inference_on_frame(img)

    if len(labels) > 0:
        top_label = int(labels[0])
        if top_label in top_label_counts:
            top_label_counts[top_label] += 1

    frame_results.append(
        {
            "frame_path": frame_path,
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }
    )

num_frames = len(frame_results)
psax_ratio = top_label_counts[2] / num_frames
hinge_ratio = top_label_counts[1] / num_frames

locked_label = None

if psax_ratio > dominant_label_threshold:
    locked_label = 2
elif hinge_ratio > dominant_label_threshold:
    locked_label = 1

print("[INFO] Video-level label summary")
print(f"        PSAX_MV top prediction ratio: {psax_ratio:.2%}")
print(f"        HINGE   top prediction ratio: {hinge_ratio:.2%}")

if locked_label is not None:
    print(
        f"[INFO] Locking detections to {CLASS_MAP[locked_label]} "
        f"(>{dominant_label_threshold:.0%} of frames)"
    )
else:
    print("[INFO] No dominant label found; keeping all detections")


assign_track_ids(
    frame_results=frame_results,
    locked_label=locked_label,
    iou_threshold=track_iou_threshold,
)


# --------------------------------------------------
# SECOND PASS: DRAW + WRITE VIDEO
# --------------------------------------------------

print("[INFO] Rendering video...")

for frame_result in tqdm(frame_results, desc="Rendering pass"):
    img = load_frame(frame_result["frame_path"])

    boxes = frame_result["render_boxes"]
    labels = frame_result["render_labels"]
    scores = frame_result["render_scores"]
    track_ids = frame_result["track_ids"]

    vis = img.copy()

    # Normalize for visualization
    if vis.dtype != np.uint8:
        vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
        vis = vis.astype(np.uint8)

    # Draw boxes first
    vis = draw_predictions(vis, boxes, labels, scores, track_ids)

    h, w = vis.shape[:2]
    panel_width = 250

    # Create extended canvas
    canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    canvas[:, :w] = vis

    # Draw panel on the right
    canvas = draw_prediction_panel(
        canvas,
        boxes,
        labels,
        scores,
        track_ids,
        image_width=w,
    )

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
