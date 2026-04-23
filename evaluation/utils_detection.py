import torch
from torchvision.ops import box_iou

import torch
from torchvision.ops import box_iou


def match_with_fp_fn(pred_boxes, pred_labels,
                     gt_boxes, gt_labels,
                     iou_threshold=0.5):
    """
    Full matching including:
    - True positives
    - False negatives
    - False positives
    """

    # Compute IoU matrix [N_pred, N_gt]
    iou_matrix = box_iou(pred_boxes, gt_boxes)

    matched_gt = []   # ground truth labels
    matched_pred = [] # predicted labels

    used_gt = set()   # track GT already matched
    used_pred = set() # track predictions already matched

    # ---------------------------
    # 1. MATCH PRED → GT
    # ---------------------------
    for pred_idx in range(len(pred_boxes)):

        # Get IoU values between this prediction and all GT boxes
        ious = iou_matrix[pred_idx]

        # Find best matching GT
        max_iou, gt_idx = torch.max(ious, dim=0)

        # If IoU is good AND GT not already used
        if max_iou >= iou_threshold and gt_idx.item() not in used_gt:

            matched_gt.append(gt_labels[gt_idx].item())      # GT class
            matched_pred.append(pred_labels[pred_idx].item())# predicted class

            used_gt.add(gt_idx.item())     # mark GT as used
            used_pred.add(pred_idx)        # mark prediction as used

    # ---------------------------
    # 2. FALSE NEGATIVES (missed GT)
    # ---------------------------
    for gt_idx in range(len(gt_boxes)):

        if gt_idx not in used_gt:

            matched_gt.append(gt_labels[gt_idx].item())  # GT exists
            matched_pred.append(0)  # predicted as background

    # ---------------------------
    # 3. FALSE POSITIVES (extra preds)
    # ---------------------------
    for pred_idx in range(len(pred_boxes)):

        if pred_idx not in used_pred:

            matched_gt.append(0)  # background
            matched_pred.append(pred_labels[pred_idx].item())

    return matched_gt, matched_pred