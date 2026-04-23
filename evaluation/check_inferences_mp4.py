from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

left_video_path = Path(
    fr"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids\inferences\PLAX_new_rule_track.mp4"
)
right_video_path = Path(
    fr"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids\inferences\inference_PLAX_best_model_new_rule.mp4"
)

left_label = "Box Track"
right_label = "Confidence Track"

resize_height = 320
window_name = "Inference MP4 Checker"
screen_width = 1800
frame_step = 10

LEFT_ARROW_KEYS = {81, 2424832, 65361}
RIGHT_ARROW_KEYS = {83, 2555904, 65363}
UP_ARROW_KEYS = {82, 2490368, 65362}
DOWN_ARROW_KEYS = {84, 2621440, 65364}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def resolve_label(video_path, custom_label):
    if custom_label is not None:
        return custom_label
    return video_path.stem


def open_video(video_path):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    return cap


def read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def resize_with_padding(frame, target_height):
    h, w = frame.shape[:2]
    scale = target_height / h
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(frame, (new_w, target_height))

    return resized


def annotate_frame(frame, label, frame_idx, total_frames):
    annotated = frame.copy()

    cv2.putText(
        annotated,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        f"Frame: {frame_idx}/{total_frames - 1}",
        (10, annotated.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def annotate_player_state(frame, paused):
    if not paused:
        return frame

    annotated = frame.copy()
    cv2.putText(
        annotated,
        "PAUSED",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def stack_side_by_side(left_frame, right_frame):
    left_h, left_w = left_frame.shape[:2]
    right_h, right_w = right_frame.shape[:2]

    max_h = max(left_h, right_h)
    canvas = np.zeros((max_h, left_w + right_w, 3), dtype=np.uint8)
    canvas[:left_h, :left_w] = left_frame
    canvas[:right_h, left_w:left_w + right_w] = right_frame

    return canvas


def maybe_scale_to_screen(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    scale = max_width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def get_playback_delay_ms(*fps_values):
    valid_fps = [fps for fps in fps_values if fps and fps > 0]
    if not valid_fps:
        return 33

    playback_fps = min(valid_fps)
    return max(1, int(round(1000 / playback_fps)))


def print_controls():
    print("[INFO] Controls")
    print("       q: quit")
    print("       p or space: pause/resume")
    print("       right arrow or d: next frame")
    print("       left arrow or a: previous frame")
    print(f"       up arrow or w: jump forward {frame_step} frames")
    print(f"       down arrow or s: jump backward {frame_step} frames")


def update_frame_index_from_key(key, current_frame, max_frames):
    if key in RIGHT_ARROW_KEYS or key == ord("d"):
        return min(current_frame + 1, max_frames - 1), True
    if key in LEFT_ARROW_KEYS or key == ord("a"):
        return max(current_frame - 1, 0), True
    if key in UP_ARROW_KEYS or key == ord("w"):
        return min(current_frame + frame_step, max_frames - 1), True
    if key in DOWN_ARROW_KEYS or key == ord("s"):
        return max(current_frame - frame_step, 0), True

    return current_frame, False


def main():
    left_cap = open_video(left_video_path)
    right_cap = open_video(right_video_path)

    try:
        left_total_frames = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        right_total_frames = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        left_fps = left_cap.get(cv2.CAP_PROP_FPS)
        right_fps = right_cap.get(cv2.CAP_PROP_FPS)
        max_frames = min(left_total_frames, right_total_frames)
        playback_delay_ms = get_playback_delay_ms(left_fps, right_fps)

        if max_frames <= 0:
            raise RuntimeError("One of the videos has no readable frames")

        left_name = resolve_label(left_video_path, left_label)
        right_name = resolve_label(right_video_path, right_label)

        print(f"[INFO] Left video:  {left_video_path}")
        print(f"[INFO] Right video: {right_video_path}")
        print(f"[INFO] Synced frames available: {max_frames}")
        print(f"[INFO] Playback delay: {playback_delay_ms} ms")
        print_controls()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 700)

        current_frame = 0
        paused = False

        while current_frame < max_frames:
            left_frame = read_frame(left_cap, current_frame)
            right_frame = read_frame(right_cap, current_frame)

            if left_frame is None or right_frame is None:
                break

            left_frame = resize_with_padding(left_frame, resize_height)
            right_frame = resize_with_padding(right_frame, resize_height)

            left_frame = annotate_frame(
                left_frame,
                left_name,
                current_frame,
                max_frames,
            )
            right_frame = annotate_frame(
                right_frame,
                right_name,
                current_frame,
                max_frames,
            )

            combined = stack_side_by_side(left_frame, right_frame)
            combined = annotate_player_state(combined, paused)
            combined = maybe_scale_to_screen(combined, screen_width)

            cv2.imshow(window_name, combined)

            wait_time = 0 if paused else playback_delay_ms
            key = cv2.waitKeyEx(wait_time)

            if key in {ord("q"), ord("Q")}:
                break
            if key in {ord("p"), ord("P"), 32}:
                paused = not paused
                continue

            current_frame, moved = update_frame_index_from_key(
                key,
                current_frame,
                max_frames,
            )
            if moved:
                continue

            if paused:
                continue

            current_frame += 1

    finally:
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
