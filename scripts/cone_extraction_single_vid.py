from pathlib import Path
import numpy as np
import pydicom
import cv2
import pandas as pd
import json

from src.preprocessing.cone_extraction import extract_cone_mask
from src.preprocessing.cone_crop import crop_to_cone


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

dicom_path = Path(rf"\\NAS3_Z\all\DB_TARTAGLIA\496_estudios_tartaglia_jmcrespi\VHIR_ECO_957372\UIDVHIR_ECO_957372\1.2.826.0.1.3680043.2.135.738651.68951324.7.1708011403.610.42\1.2.826.0.1.3680043.2.135.738651.68951324.7.1708011437.454.7.dcm")  # <-- CHANGE THIS
output_dir = Path(rf"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_test_vids")

target_size = (224, 224)  # (H, W)

save_npy = True
save_video = True


# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

frames_dir = output_dir / "frames_A2C_feo"
frames_dir.mkdir(parents=True, exist_ok=True)

failure_log = []


# --------------------------------------------------
# LOAD DICOM
# --------------------------------------------------

dcm = pydicom.dcmread(dicom_path)
frames = dcm.pixel_array  # expected (T, H, W) or (T, H, W, C)

num_frames = frames.shape[0]

print(f"[INFO] Total frames: {num_frames}")


# --------------------------------------------------
# EXTRACT FPS FROM DICOM
# --------------------------------------------------

def get_fps(dcm):
    if hasattr(dcm, "RecommendedDisplayFrameRate"):
        return float(dcm.RecommendedDisplayFrameRate)

    if hasattr(dcm, "FrameTime"):
        return 1000.0 / float(dcm.FrameTime)

    if hasattr(dcm, "CineRate"):
        return float(dcm.CineRate)

    print("[WARNING] FPS not found → using default = 20")
    return 20.0


fps = get_fps(dcm)
print(f"[INFO] FPS: {fps:.2f}")

metadata = {
    "dicom_path": str(dicom_path),
    "num_frames": int(num_frames),
    "fps": float(fps),
    "frame_shape": list(frames.shape),
    "target_size": list(target_size),
    "patient_id": getattr(dcm, "PatientID", None),
    "study_uid": getattr(dcm, "StudyInstanceUID", None),
    "series_uid": getattr(dcm, "SeriesInstanceUID", None),
    "modality": getattr(dcm, "Modality", None)
}

with open(frames_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)


# --------------------------------------------------
# EXTRACT CONE MASK (ONCE)
# --------------------------------------------------

print("[INFO] Extracting cone mask...")

mask = extract_cone_mask(frames)

mask_area = mask.sum()
image_area = mask.shape[0] * mask.shape[1]
area_ratio = mask_area / image_area

print(f"[INFO] Cone mask area ratio: {area_ratio:.4f}")

if mask_area < 500 or area_ratio < 0.05 or area_ratio > 0.9:
    raise RuntimeError("Invalid cone mask → aborting")


# --------------------------------------------------
# VIDEO WRITER
# --------------------------------------------------

if save_video:
    video_writer = cv2.VideoWriter(
        str(output_dir / "preprocessed_video_psax_mv.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (target_size[1], target_size[0])  # (W, H)
    )


# --------------------------------------------------
# MAIN PROCESSING LOOP
# --------------------------------------------------

print("[INFO] Processing frames...")

for frame_idx in range(num_frames):

    try:
        frame = frames[frame_idx]

        # ------------------------------------------
        # FIX: Ensure channel dimension
        # ------------------------------------------
        if frame.ndim == 2:
            frame = frame[..., None]

        # ------------------------------------------
        # CROP CONE
        # ------------------------------------------
        cropped_img, cropped_mask, _ = crop_to_cone(
            frame,
            mask,
            margin=10
        )

        # Ensure shape consistency
        if cropped_img.ndim == 2:
            cropped_img = cropped_img[..., None]

        # Apply mask
        cropped_img = cropped_img * cropped_mask[..., None]

        # ------------------------------------------
        # VALIDATION
        # ------------------------------------------
        h, w = cropped_img.shape[:2]

        if h < 50 or w < 50:
            failure_log.append({
                "frame": frame_idx,
                "reason": "small_crop"
            })
            continue

        # ------------------------------------------
        # RESIZE
        # ------------------------------------------
        resized_img = cv2.resize(
            cropped_img,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

        if resized_img.ndim == 2:
            resized_img = resized_img[..., None]

        # ------------------------------------------
        # SAVE NPY
        # ------------------------------------------
        if save_npy:
            np.save(
                frames_dir / f"frame_{frame_idx:04d}.npy",
                resized_img
            )

        # ------------------------------------------
        # SAVE VIDEO FRAME
        # ------------------------------------------
        if save_video:
            vis = resized_img

            # Normalize to uint8 if needed
            if vis.dtype != np.uint8:
                vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
                vis = vis.astype(np.uint8)

            # Ensure 3 channels for video
            if vis.ndim == 2:
                vis = np.stack([vis]*3, axis=-1)
            elif vis.shape[2] == 1:
                vis = np.repeat(vis, 3, axis=2)

            video_writer.write(vis)

    except Exception as e:
        failure_log.append({
            "frame": frame_idx,
            "reason": str(e)
        })


# --------------------------------------------------
# FINALIZE
# --------------------------------------------------

if save_video:
    video_writer.release()


# Save failure log
if len(failure_log) > 0:
    failure_df = pd.DataFrame(failure_log)
    failure_df.to_csv(output_dir / "failure_log.csv", index=False)


print("\n--------------------------------------")
print("PROCESSING SUMMARY")
print("--------------------------------------")
print(f"Total frames: {num_frames}")
print(f"Failures: {len(failure_log)}")
print(f"Output dir: {output_dir}")
print("--------------------------------------")