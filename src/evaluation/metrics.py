import torch
from torchvision.ops import box_iou


def calculate_label_precision(pred_labels, true_labels):
    if len(pred_labels) == 0:
        return 0.0
    # sort both independently — order doesn't matter, content does
    pred_sorted = pred_labels.sort().values
    true_sorted = true_labels.sort().values
    # compare up to the shorter length
    n = min(len(pred_sorted), len(true_sorted))
    correct = (pred_sorted[:n] == true_sorted[:n]).sum().item()
    return correct / len(pred_labels)

def calculate_match_precision(pred_boxes, true_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return 0.0

    iou = box_iou(pred_boxes, true_boxes)  # [N_pred, N_gt]

    # each predicted box gets credit for its BEST matching gt box only
    best_iou_per_pred = iou.max(dim=1).values  # [N_pred]
    tp = (best_iou_per_pred > iou_threshold).sum().item()

    return tp / len(pred_boxes)