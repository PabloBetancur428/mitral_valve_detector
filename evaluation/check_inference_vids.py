import cv2
import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
video_paths = [
    r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids\inferences\inference_a2c_feo_t05.mp4",
    r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids\inferences\inference_a2c_feo_t05_2.mp4",
    r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids\inferences\inference_a2c_feo_t02_2.mp4",
]

labels = [
    "Model 1",
    "Model 2",
    "Model 3",
]

resize_height = 224
window_name = "Multi-Video Frame Viewer"

# layout: "horizontal" or "grid"
layout_mode = "horizontal"
grid_cols = 2  # used only if layout_mode == "grid"

screen_width = 1600  # max display width


# --------------------------------------------------
# LOAD VIDEOS
# --------------------------------------------------
caps = [cv2.VideoCapture(p) for p in video_paths]

if not all([cap.isOpened() for cap in caps]):
    raise RuntimeError("Error opening one or more videos")

num_videos = len(caps)

num_frames_list = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
max_frames = min(num_frames_list)

print(f"[INFO] Videos loaded: {num_videos}")
print(f"[INFO] Frames available (synced): {max_frames}")


# --------------------------------------------------
# WINDOW SETUP (IMPORTANT FIX)
# --------------------------------------------------
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 600)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def resize_frame(frame, target_size=224):
    h, w = frame.shape[:2]

    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    # Create square canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Center the image
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def stack_grid(frames, cols=2):
    rows = []
    for i in range(0, len(frames), cols):
        row_frames = frames[i:i+cols]

        # If last row incomplete → pad with black images
        if len(row_frames) < cols:
            h, w = row_frames[0].shape[:2]
            for _ in range(cols - len(row_frames)):
                row_frames.append(np.zeros((h, w, 3), dtype=np.uint8))

        row = np.hstack(row_frames)
        rows.append(row)

    return np.vstack(rows)


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
current_frame = 0

while True:
    frames = []

    for i, cap in enumerate(caps):
        frame = read_frame(cap, current_frame)

        if frame is None:
            frame = np.zeros((resize_height, resize_height, 3), dtype=np.uint8)

        frame = resize_frame(frame, target_size=224)

        # Label overlay
        label = labels[i] if i < len(labels) else f"Video {i}"
        cv2.putText(
            frame,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        frames.append(frame)

    # --------------------------------------------------
    # LAYOUT
    # --------------------------------------------------
    if layout_mode == "horizontal":
        combined = np.hstack(frames)
    elif layout_mode == "grid":
        combined = stack_grid(frames, cols=grid_cols)
    else:
        raise ValueError("Invalid layout_mode")

    # --------------------------------------------------
    # AUTO SCALE TO SCREEN (CRITICAL FIX)
    # --------------------------------------------------
    h, w = combined.shape[:2]

    if w > screen_width:
        scale = screen_width / w
        combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

    # --------------------------------------------------
    # FRAME INFO OVERLAY
    # --------------------------------------------------
    cv2.putText(
        combined,
        f"Frame: {current_frame}/{max_frames - 1}",
        (10, combined.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------
    # SHOW
    # --------------------------------------------------
    cv2.imshow(window_name, combined)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == 83 or key == ord('d'):  # right arrow or 'd'
        current_frame = min(current_frame + 1, max_frames - 1)
    elif key == 81 or key == ord('a'):  # left arrow or 'a'
        current_frame = max(current_frame - 1, 0)
    elif key == ord('w'):  # jump forward
        current_frame = min(current_frame + 10, max_frames - 1)
    elif key == ord('s'):  # jump backward
        current_frame = max(current_frame - 10, 0)


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------
for cap in caps:
    cap.release()

cv2.destroyAllWindows()