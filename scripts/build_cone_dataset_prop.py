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


# Temporal propagation configuration
OFFSETS_MAP = {
    "A2C": [-5,-3, 0, 3, 5],
    "A3C": [-5, -3, 0, 3, 5],
    "PLAX": [-5, -3, 0, 3, 5],
    "PSAX_MV": [-5, -3, 0, 3, 5],
    "A4C": [0, 3, 5]  # less augmentation for dominant class
}


def is_valid_sample(mask, cropped_img, bboxes):
    """
    Validate whether a processed sample is usable.

    This function prevents:
    - bad cone extraction
    - broken crops
    - invalid bounding boxes

    Returns
    -------
    bool
        True if sample is valid, False otherwise
    """

    # --------------------------------------------------
    # 1. Mask validation
    # --------------------------------------------------

    mask_area = mask.sum()
    image_area = mask.shape[0] * mask.shape[1]

    if mask_area < 500:
        return False

    area_ratio = mask_area / image_area

    if area_ratio < 0.05 or area_ratio > 0.9:
        return False


    # --------------------------------------------------
    # 2. Crop validation
    # --------------------------------------------------

    h, w = cropped_img.shape[:2]

    if h < 50 or w < 50:
        print(f"[FAIL][CROP] too small → h={h}, w={w}")
        return False


    # --------------------------------------------------
    # 3. Bounding box validation
    # --------------------------------------------------

    for box in bboxes:

        x1, y1, x2, y2 = box

        # invalid geometry
        if x2 <= x1 or y2 <= y1:
            print(f"[FAIL][BBOX_GEOMETRY] box={box}")
            return False

        # completely خارج image
        if x2 < 0 or y2 < 0:
            print(f"[FAIL][BBOX_NEGATIVE] box={box}")
            return False

        if x1 > w or y1 > h:
            print(f"[FAIL][BBOX_OUTSIDE] box={box} img_shape=({h},{w})")
            return False

        # extremely small bbox → likely wrong
        box_area = (x2 - x1) * (y2 - y1)
        img_area = h * w

        if box_area / img_area < 0.0005:
            print(f"[FAIL][BBOX_TOO_SMALL] box={box} ratio={box_area/img_area:.6f}")
            return False

    return True

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

def log_failure(failure_log, reason, fname, study_name, frame_idx=None, offset=None, extra_info=None):
    """
    Store a failure entry in a structured way.

    Parameters
    ----------
    failure_log : list
        Global list collecting all failures

    reason : str
        Type of failure (e.g. CONE_FAIL, BBOX_FAIL, etc.)

    fname : str
        DICOM filename

    study_name : str
        Study identifier

    frame_idx : int, optional
        Frame index involved

    offset : int, optional
        Offset applied

    extra_info : str, optional
        Additional debug info
    """

    failure_log.append({
        "reason": reason,             # type of failure
        "fname": fname,              # file name
        "study_id": study_name,      # study identifier
        "frame_index": frame_idx,    # frame involved
        "offset": offset,            # offset used
        "extra_info": extra_info     # additional debug message
    })


if __name__ == '__main__':
    # Data path
    annotations_csv = Path(
            r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\AoVdetector\data\annotations_boxes\mitral_annotations_2502.csv"
        )

    dataset_root = Path(
            r"\\NAS3_Z\all\DB_TARTAGLIA\496_estudios_tartaglia_jmcrespi"
        )

    output_root = Path(
        r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco"
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

    # Take some samples
    #df = df.sample(30, random_state=42)

    # For debugging
    #df = df.head(10)

    #df = df[df['view'] == 'PSAX_MV'].reset_index(drop=True)
    #df = df.head(5)
    print(f"Total annotated frames: {len(df)}")


    #### Index all DICOM

    print("Indexing DICOM files...")

    dicom_index = {}
    new_rows = []
    failed_rows = []

    # rglob to search all subdirectories - Lookup O(1) to avoid enteing NAS every time - Check all the dcm file in "DB_TARTAGLIA"
    for dcm_path in dataset_root.rglob("*.dcm"):
        dicom_index[dcm_path.name] = dcm_path

    print(f"Total DICOM files indexed in TARTAGLIA: {len(dicom_index)}")

    failure_log = []

    failed_samples = 0
    cone_failures = 0
    index_failures = 0
    bbox_failures = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        try:
            # Extract video info
            fname = row["fname"]
            study_name = row["uid_study"]
            view = row["view"]
            print(f"\n[START] study={study_name} file={fname} view={view} base_frame={row['frame_index']}")
            dicom_path = dicom_index[fname]

            dcm = pydicom.dcmread(dicom_path)
            frames = dcm.pixel_array

            mask = extract_cone_mask(frames)

            # Validate mask
            mask_area = mask.sum()
            image_area = mask.shape[0] * mask.shape[1]
            area_ratio = mask_area / image_area

            print(f"[CONE] mask_area={mask_area} area_ratio={area_ratio:.4f}")

            if mask_area < 500 or area_ratio < 0.05 or area_ratio > 0.9:

                cone_failures += 1

                log_failure(
                    failure_log,
                    reason="CONE_FAIL",
                    fname=fname,
                    study_name=study_name,
                    frame_idx=row["frame_index"],
                    extra_info=f"mask_area={mask_area}, ratio={area_ratio:.4f}"
                )

                continue

            base_frame_index = int(row["frame_index"])

            # Check what's being processedv
            # --------------------------------------------------
            # Sanity check: frame index
            # --------------------------------------------------

            if base_frame_index >= frames.shape[0]:

                index_failures += 1

                log_failure(
                    failure_log,
                    reason="FRAME_INDEX_FAIL",
                    fname=fname,
                    study_name=study_name,
                    frame_idx=base_frame_index,
                    extra_info=f"num_frames={frames.shape[0]}"
                )

                continue


            # Create offsets
            offsets = OFFSETS_MAP.get(view, [0])
            num_frames = frames.shape[0]

            for offset in offsets:
                
                # Circular indexing just in case
                new_frame_index = (base_frame_index + offset) % num_frames
                frame = frames[new_frame_index]
                print(f"[OFFSET] offset={offset} → new_frame={new_frame_index}")


                # --------------------------------------------------
                # Crop cone
                # --------------------------------------------------

                cropped_img, cropped_mask, crop_origin = crop_to_cone(
                    frame,
                    mask,
                    margin=10
                )

                h, w = cropped_img.shape[:2]
                print(f"[CROP] shape=({h},{w}) origin={crop_origin}")
                # Apply mask
                cropped_img = cropped_img * cropped_mask[..., None]

            # if mask.sum() < 50:

            #     cone_failures += 1

            #     print(f"[CONE ERROR] {fname} frame {frame_index} empty mask")

            #     np.save(
            #         failed_dir / f"{Path(fname).stem}_frame{frame_index}.npy",
            #         frame
            #     )

            #     continue




                # --------------------------------------------------
                # Bounding boxes
                # --------------------------------------------------

                bboxes = [
                    [row.box1_x1, row.box1_y1, row.box1_x2, row.box1_y2],
                    [row.box2_x1, row.box2_y1, row.box2_x2, row.box2_y2]
                ]

                shifted_boxes = shift_bboxes(bboxes, crop_origin)

                if not is_valid_sample(mask, cropped_img, shifted_boxes):

                    bbox_failures += 1

                    log_failure(
                        failure_log,
                        reason="INVALID_SAMPLE",
                        fname=fname,
                        study_name=study_name,
                        frame_idx=new_frame_index,
                        offset=offset
                    )

                    continue


                # --------------------------------------------------
                # Resize image and boxes
                # --------------------------------------------------

                resized_img, resized_boxes = resize_image_and_bboxes(
                    cropped_img,
                    shifted_boxes,
                    target_size
                )

                # Check that resizing went well
                valid = True
                for x1, y1, x2, y2 in resized_boxes:
                    if x1 < 0 or y1 < 0 or x2 > target_size[1] or y2 > target_size[0]:
                        valid = False
                        break

                if not valid:

                    log_failure(
                        failure_log,
                        reason="RESIZE_OUT_OF_BOUNDS",
                        fname=fname,
                        study_name=study_name,
                        frame_idx=new_frame_index,
                        offset=offset
                    )

                    continue


                # --------------------------------------------------
                # Assign class labels
                # --------------------------------------------------
                labels = assign_classes(view, resized_boxes)


                # --------------------------------------------------
                # Save processed image
                # --------------------------------------------------

                fname_stem = Path(fname).stem

                sample_id = f"{study_name}_{fname_stem}_frame{new_frame_index}_offset{offset}"

                img_path = imgs_dir / f"{sample_id}.npy"

                np.save(img_path, resized_img)


                new_rows.append({
                    "processed_image": f"{sample_id}.npy",

                    "box1_x1": resized_boxes[0][0],
                    "box1_y1": resized_boxes[0][1],
                    "box1_x2": resized_boxes[0][2],
                    "box1_y2": resized_boxes[0][3],

                    "box2_x1": resized_boxes[1][0],
                    "box2_y1": resized_boxes[1][1],
                    "box2_x2": resized_boxes[1][2],
                    "box2_y2": resized_boxes[1][3],

                    "label1": labels[0],
                    "label2": labels[1],

                    "view": view,
                    "frame_index": new_frame_index,
                    "offset": offset,
                    "study_id": study_name,
                    "fname": fname
                })


        except Exception as e:

            failed_samples += 1

            log_failure(
                failure_log,
                reason="EXCEPTION",
                fname=fname if 'fname' in locals() else "UNKNOWN",
                study_name=study_name if 'study_name' in locals() else "UNKNOWN",
                frame_idx=base_frame_index if 'base_frame_index' in locals() else None,
                offset=offset if 'offset' in locals() else None,
                extra_info=str(e)  # store actual error message
            )

            print(f"[ERROR] → {e}")

            continue


    new_df = pd.DataFrame(new_rows)
    updated_csv_path = output_root / "annotations_propagated_aSaco.csv"
    new_df.to_csv(updated_csv_path, index=False)

    # --------------------------------------------------
    # Save failure log
    # --------------------------------------------------

    failure_df = pd.DataFrame(failure_log)

    failure_csv_path = output_root / "failure_log_aSaco.csv"
    failure_txt_path = output_root / "failure_log_aSaco.txt"

    # Save as CSV (structured, best for analysis)
    failure_df.to_csv(failure_csv_path, index=False)

    # Save as TXT (human readable)
    with open(failure_txt_path, "w") as f:
        for entry in failure_log:
            f.write(str(entry) + "\n")

    print(f"Failure log saved to: {failure_csv_path}")

    print(f"Updated annotations saved to: {updated_csv_path}")
    print("Total failed cone extractioned: ", failed_samples)

    print("\n--------------------------------------")
    print("PREPROCESSING SUMMARY")
    print("--------------------------------------")
    print("Total samples:", len(df))
    print("General failures:", failed_samples)
    print("Cone failures:", cone_failures)
    print("Frame index failures:", index_failures)

    print("BBox / validation failures:", bbox_failures)
    print("Total logged failures:", len(failure_log))   
    print("--------------------------------------")
