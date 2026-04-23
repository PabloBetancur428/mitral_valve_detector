"""
cone_crop.py

Utilities for cropping echocardiography images around the ultrasound cone
and updating bounding boxes accordingly.

The goal is to preserve the original pixel scale while removing irrelevant
background regions outside the cone.

Pipeline:

cone mask
    ↓
compute cone bounding box
    ↓
crop image to cone
    ↓
shift bounding boxes
"""

# NumPy is required for numerical operations and mask processing
import numpy as np


# ---------------------------------------------------------
# STEP 1 — Compute cone bounding box
# ---------------------------------------------------------

def get_cone_bbox(mask):
    """
    Compute the tight bounding box of the cone mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask of the ultrasound cone.

        Shape:
            (H, W)

    Returns
    -------
    x_min, y_min, x_max, y_max : int
        Coordinates of the cone bounding box.
    """

    # np.where returns indices of pixels where condition is True
    # Here we locate all pixels belonging to the cone
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:

        raise ValueError("Cone mask is empty — cone detection failed.")

    # Compute minimum x coordinate of cone pixels
    x_min = xs.min()

    # Compute maximum x coordinate
    x_max = xs.max()

    # Compute minimum y coordinate
    y_min = ys.min()

    # Compute maximum y coordinate
    y_max = ys.max()

    # Return bounding box coordinates
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------
# STEP 2 — Crop image and mask to cone region
# ---------------------------------------------------------

def crop_to_cone(image, mask, margin=5):
    """
    Crop image and mask around the ultrasound cone.

    Parameters
    ----------
    image : np.ndarray
        Input image.

        Shape:
            (H, W, C)

    mask : np.ndarray
        Binary cone mask.

        Shape:
            (H, W)

    margin : int
        Extra pixels added around the cone bounding box.

    Returns
    -------
    cropped_image : np.ndarray
    cropped_mask : np.ndarray
    crop_origin : tuple
        (x0, y0) origin of the crop in the original image.
    """

    # Compute cone bounding box
    x_min, y_min, x_max, y_max = get_cone_bbox(mask)

    # Expand bounding box slightly using margin
    # max() ensures we don't go outside image boundary
    x_min = max(0, x_min - margin)

    # Same logic for y coordinate
    y_min = max(0, y_min - margin)

    # min() ensures we don't exceed image width
    x_max = min(image.shape[1], x_max + margin)

    # min() ensures we don't exceed image height
    y_max = min(image.shape[0], y_max + margin)

    # Crop image using numpy slicing
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Crop mask using same coordinates
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # Store crop origin for bbox shifting
    crop_origin = (x_min, y_min)

    # Return cropped data
    return cropped_image, cropped_mask, crop_origin


# ---------------------------------------------------------
# STEP 3 — Shift bounding boxes after cropping
# ---------------------------------------------------------

def shift_bboxes(bboxes, crop_origin):
    """
    Shift bounding boxes after cropping.

    Parameters
    ----------
    bboxes : list of lists
        Bounding boxes in format:

        [x1, y1, x2, y2]

    crop_origin : tuple
        (x0, y0) origin of crop.

    Returns
    -------
    shifted_bboxes : list
        Bounding boxes updated to new coordinate system.
    """

    # Extract crop origin
    x0, y0 = crop_origin

    # Create empty list for shifted boxes
    shifted = []

    # Iterate over each bounding box
    for x1, y1, x2, y2 in bboxes:

        # Shift coordinates relative to crop origin
        shifted.append([
            x1 - x0,
            y1 - y0,
            x2 - x0,
            y2 - y0
        ])

    # Return updated bounding boxes
    return shifted