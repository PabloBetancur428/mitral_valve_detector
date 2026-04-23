"""
    Code to generate a new dataset where the frames of  interest are cone extracted.

"""

from pathlib import Path  # modern path handling
import pandas as pd       # reading annotation CSV
import numpy as np        # numerical operations
import pydicom            # reading DICOM files
from tqdm import tqdm     # progress bar
from src.preprocessing.cone_extraction import extract_cone_mask
from src.preprocessing.cone_crop import get_cone_bbox, crop_to_cone, shift_bboxes
import cv2

def resize_image_and_bboxes(image, bboxes, target_size):
    """
    Resize image and scale bounding boxes accordingly.

    Parameters
    ----------
    image : np.ndarray
        Image array with shape (H, W, C)

    bboxes : list
        List of bounding boxes in format:
        [x1, y1, x2, y2]

    target_size : tuple
        Desired output size (target_height, target_width)

    Returns
    -------
    resized_image : np.ndarray

    resized_bboxes : list
    """

    # --------------------------------------------------
    # Extract original dimensions
    # --------------------------------------------------

    original_h, original_w = image.shape[:2]

    target_h, target_w = target_size


    # --------------------------------------------------
    # Compute scaling factors
    # --------------------------------------------------

    scale_x = target_w / original_w
    scale_y = target_h / original_h


    # --------------------------------------------------
    # Resize image
    # --------------------------------------------------

    resized_image = cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------
    # Scale bounding boxes
    # --------------------------------------------------

    resized_bboxes = []

    for x1, y1, x2, y2 in bboxes:

        new_x1 = x1 * scale_x
        new_y1 = y1 * scale_y
        new_x2 = x2 * scale_x
        new_y2 = y2 * scale_y

        resized_bboxes.append([
            new_x1,
            new_y1,
            new_x2,
            new_y2
        ])


    return resized_image, resized_bboxes

# ---------------------------------------------------------
# Assign class labels according to view and bbox position
# ---------------------------------------------------------

def assign_classes(view, bboxes):
    """
    Determine class labels based on view and bounding box positions.

    Parameters
    ----------
    view : str
        Echocardiography view.

    bboxes : list
        List of bounding boxes [[x1,y1,x2,y2], [x1,y1,x2,y2]]

    Returns
    -------
    labels : list
        List of class IDs corresponding to each bbox.
    """
    CLASS_MAP = {
        # A2C
        "A2C_hinge_left": 1,
        "A2C_hinge_right": 2,

        # A3C
        "A3C_hinge_left": 3,
        "A3C_hinge_right": 4,

        # A4C
        "A4C_hinge_left": 5,
        "A4C_hinge_right": 6,

        # PSAX
        "PSAX_MV_mitral_valve": 7,

        # PLAX
        "PLAX_hinge_bottom": 8,
        "PLAX_hinge_top": 9
    }


    # --------------------------------------------------
    # Compute bounding box centers
    # --------------------------------------------------

    centers = []

    for x1, y1, x2, y2 in bboxes:

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        centers.append((cx, cy))


    # --------------------------------------------------
    # PSAX_MV (single anatomical structure)
    # --------------------------------------------------

    if view == "PSAX_MV":

        return [
            CLASS_MAP["PSAX_MV_mitral_valve"],
            CLASS_MAP["PSAX_MV_mitral_valve"]
        ]


    # --------------------------------------------------
    # A2C and A4C (left / right ordering)
    # --------------------------------------------------

    if view in ["A2C", "A3C", "A4C"]:

        # sort indices by x coordinate
        sorted_idx = sorted(range(2), key=lambda i: centers[i][0])

        labels = [None, None]

        if view == "A2C":

            labels[sorted_idx[0]] = CLASS_MAP["A2C_hinge_left"]
            labels[sorted_idx[1]] = CLASS_MAP["A2C_hinge_right"]

        elif view == "A3C":
            labels[sorted_idx[0]] = CLASS_MAP["A3C_hinge_left"]
            labels[sorted_idx[1]] = CLASS_MAP["A3C_hinge_right"]

        elif view == "A4C":

            labels[sorted_idx[0]] = CLASS_MAP["A4C_hinge_left"]
            labels[sorted_idx[1]] = CLASS_MAP["A4C_hinge_right"]

        return labels


    # --------------------------------------------------
    # A3C and PLAX (top / bottom ordering)
    # --------------------------------------------------

    if view in ["PLAX"]:

        # sort indices by y coordinate
        sorted_idx = sorted(range(2), key=lambda i: centers[i][1])

        labels = [None, None]

        if view == "PLAX":

            labels[sorted_idx[0]] = CLASS_MAP["PLAX_hinge_top"]
            labels[sorted_idx[1]] = CLASS_MAP["PLAX_hinge_bottom"]

        return labels


    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    return [0, 0]


if __name__ == '__main__':
    # Data path
    annotations_csv = Path(
            r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\AoVdetector\data\annotations_boxes\mitral_annotations_2502.csv"
        )

    dataset_root = Path(
            r"\\NAS3_Z\all\DB_TARTAGLIA\496_estudios_tartaglia_jmcrespi"
        )

    output_root = Path(
        r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset"
    )

    

    # Set a target size for resizing
    target_size = (224, 224) #  h, w
    # Subdirs to store results
    imgs_dir = output_root / "images"
    masks_dir = output_root / "masks"
    failed_dir = output_root / "failed_frames"

    imgs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(exist_ok=True)

    # Load csv
    df = pd.read_csv(annotations_csv)

    # remove discarded studies
    df = df[df["frame_index"] != -1]

    # Reset inder
    df = df.reset_index(drop=True)

    # For debugging
    #df = df.head(5)

    #df = df[df['view'] == 'PSAX_MV'].reset_index(drop=True)
    #df = df.head(5)
    print(f"Total annotated frames: {len(df)}")


    #### Index all DICOM

    print("Indexing DICOM files...")

    dicom_index = {}

    # rglob to search all subdirectories - Lookup O(1) to avoid enteing NAS every time - Check all the dcm file in "DB_TARTAGLIA"
    for dcm_path in dataset_root.rglob("*.dcm"):
        dicom_index[dcm_path.name] = dcm_path

    print(f"Total DICOM files indexed in TARTAGLIA: {len(dicom_index)}")

    failed_samples = 0
    cone_failures = 0
    index_failures = 0
    bbox_failures = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        try:

            fname = row["fname"]
            study_name = row["uid_study"]

            dicom_path = dicom_index[fname]

            dcm = pydicom.dcmread(dicom_path)
            frames = dcm.pixel_array

            frame_index = int(row["frame_index"])

            # --------------------------------------------------
            # Sanity check: frame index
            # --------------------------------------------------

            if frame_index >= frames.shape[0]:

                index_failures += 1

                print(f"[FRAME ERROR] {fname} frame {frame_index} exceeds {frames.shape[0]}")

                continue

            frame = frames[frame_index]


            # --------------------------------------------------
            # Cone extraction
            # --------------------------------------------------

            mask = extract_cone_mask(frames)

            if mask.sum() < 50:

                cone_failures += 1

                print(f"[CONE ERROR] {fname} frame {frame_index} empty mask")

                np.save(
                    failed_dir / f"{Path(fname).stem}_frame{frame_index}.npy",
                    frame
                )

                continue


            # --------------------------------------------------
            # Crop cone
            # --------------------------------------------------

            cropped_img, cropped_mask, crop_origin = crop_to_cone(
                frame,
                mask,
                margin=10
            )
            cropped_img = cropped_img * cropped_mask[..., None]

            # --------------------------------------------------
            # Bounding boxes
            # --------------------------------------------------

            bboxes = [
                [row.box1_x1, row.box1_y1, row.box1_x2, row.box1_y2],
                [row.box2_x1, row.box2_y1, row.box2_x2, row.box2_y2]
            ]

            shifted_boxes = shift_bboxes(bboxes, crop_origin)


            # --------------------------------------------------
            # Resize image and boxes
            # --------------------------------------------------

            resized_img, resized_boxes = resize_image_and_bboxes(
                cropped_img,
                shifted_boxes,
                target_size
            )


            # --------------------------------------------------
            # Assign class labels
            # --------------------------------------------------

            view = row["view"]

            labels = assign_classes(view, resized_boxes)


            # --------------------------------------------------
            # Save processed image
            # --------------------------------------------------

            fname_stem = Path(fname).stem

            sample_id = f"{study_name}_{fname_stem}_frame{frame_index}"

            img_path = imgs_dir / f"{sample_id}.npy"

            np.save(img_path, resized_img)


            # --------------------------------------------------
            # Update dataframe
            # --------------------------------------------------

            df.loc[idx, "processed_image"] = f"{sample_id}.npy"

            df.loc[idx, "box1_x1"] = resized_boxes[0][0]
            df.loc[idx, "box1_y1"] = resized_boxes[0][1]
            df.loc[idx, "box1_x2"] = resized_boxes[0][2]
            df.loc[idx, "box1_y2"] = resized_boxes[0][3]

            df.loc[idx, "box2_x1"] = resized_boxes[1][0]
            df.loc[idx, "box2_y1"] = resized_boxes[1][1]
            df.loc[idx, "box2_x2"] = resized_boxes[1][2]
            df.loc[idx, "box2_y2"] = resized_boxes[1][3]

            df.loc[idx, "label1"] = labels[0]
            df.loc[idx, "label2"] = labels[1]


        except Exception as e:

            failed_samples += 1

            print(
                f"[GENERAL ERROR] study={study_name} file={fname} frame={frame_index} → {e}"
            )

            continue

    updated_csv_path = output_root / "annotations_cropped_1603_before_leaving.csv"
    df.to_csv(updated_csv_path, index=False)

    print(f"Updated annotations saved to: {updated_csv_path}")
    print("Total failed cone extractioned: ", failed_samples)

    print("\n--------------------------------------")
    print("PREPROCESSING SUMMARY")
    print("--------------------------------------")
    print("Total samples:", len(df))
    print("General failures:", failed_samples)
    print("Cone failures:", cone_failures)
    print("Frame index failures:", index_failures)
    print("--------------------------------------")
