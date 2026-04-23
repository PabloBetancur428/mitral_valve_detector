import torch
from torchvision.ops import box_iou
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.datasets.echo_dataset import EchoDataset
from src.training.dataloader import build_dataloader
from src.models.faster_rcnn_model import build_faster_rcnn_model

def match_detections(pred_boxes, pred_scores, gt_boxes, iou_threshold):
    """
    Perform matching for ONE class.

    Returns:
        tp: list of 0/1
        fp: list of 0/1
        scores: list of confidence scores
    """

    # Sort predictions by confidence (descending)
    sorted_idx = torch.argsort(pred_scores, descending=True)

    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]

    # Track matched GTs (1-to-1 matching)
    matched_gt = set()

    tp = []
    fp = []

    # Loop over predictions
    for i in range(len(pred_boxes)):

        pred_box = pred_boxes[i].unsqueeze(0)

        if len(gt_boxes) == 0:
            # No GT → everything is FP
            tp.append(0)
            fp.append(1)
            continue

        # Compute IoU with all GT
        ious = box_iou(pred_box, gt_boxes).squeeze(0)

        max_iou, gt_idx = torch.max(ious, dim=0)

        # Check match
        if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
            tp.append(1)
            fp.append(0)
            matched_gt.add(gt_idx.item())
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp, pred_scores.tolist()

def plot_pr_curve(precision, recall, class_id, iou_threshold):
    plt.plot(recall, precision, label=f"Class {class_id}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve @ IoU={iou_threshold}")
    plt.legend()
    plt.grid(True)

def compute_precision_recall(tp, fp, num_gt):
    """
    Compute PR curve from TP/FP lists.
    """

    tp = np.array(tp)
    fp = np.array(fp)

    # Cumulative sums
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    # Precision and Recall
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    recall = tp_cum / (num_gt + 1e-6)

    return precision, recall

def compute_ap(precision, recall):
    """
    Compute AP using interpolation.
    """

    # Append boundary values
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    # Make precision monotonically decreasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Compute area
    indices = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap

def evaluate_map(model, dataloader, device, num_classes, iou_threshold=0.5):

    model.eval()

    # Store per-class data
    all_tp = {c: [] for c in range(1, num_classes)}
    all_fp = {c: [] for c in range(1, num_classes)}
    all_scores = {c: [] for c in range(1, num_classes)}
    num_gt_per_class = {c: 0 for c in range(1, num_classes)}

    with torch.no_grad():
        for images, targets in dataloader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):

                pred_boxes = outputs[i]["boxes"].cpu()
                pred_labels = outputs[i]["labels"].cpu()
                pred_scores = outputs[i]["scores"].cpu()

                gt_boxes = targets[i]["boxes"].cpu()
                gt_labels = targets[i]["labels"].cpu()

                view = targets[i]["view"]

                # ---- KEEP YOUR GT FIX ----
                if view == "PSAX_MV":
                    gt_boxes = gt_boxes[:1]
                    gt_labels = gt_labels[:1]

                else:
                    gt_boxes = gt_boxes[:2]
                    gt_labels = gt_labels[:2]

                # ---- NO TOP-K HERE ----

                # Process per class
                for c in range(1, num_classes):

                    pred_mask = pred_labels == c
                    gt_mask = gt_labels == c

                    c_pred_boxes = pred_boxes[pred_mask]
                    c_pred_scores = pred_scores[pred_mask]
                    c_gt_boxes = gt_boxes[gt_mask]

                    num_gt_per_class[c] += len(c_gt_boxes)

                    tp, fp, scores = match_detections(
                        c_pred_boxes,
                        c_pred_scores,
                        c_gt_boxes,
                        iou_threshold
                    )

                    all_tp[c].extend(tp)
                    all_fp[c].extend(fp)
                    all_scores[c].extend(scores)

    # ---- COMPUTE AP ----
    ap_per_class = {}
    pr_curves = {}

    for c in range(1, num_classes):

        scores = np.array(all_scores[c])
        tp = np.array(all_tp[c])
        fp = np.array(all_fp[c])

        # ---- ADD THIS BLOCK HERE ----
        if num_gt_per_class[c] == 0 or len(scores) == 0:
            ap_per_class[c] = 0.0
            pr_curves[c] = (np.array([0]), np.array([0]))
            continue
        # --------------------------------

        # Sort globally by score
        sorted_idx = np.argsort(-scores)

        tp = tp[sorted_idx]
        fp = fp[sorted_idx]

        precision, recall = compute_precision_recall(
            tp, fp, num_gt_per_class[c]
        )

        pr_curves[c] = (precision, recall)

        ap = compute_ap(precision, recall)

        ap_per_class[c] = ap

    # ---- mAP ----
    mAP = np.mean(list(ap_per_class.values()))
    print(f"Avg precision class 1: {ap_per_class[1]:.2f}")
    print(f"Avg precision class 2: {ap_per_class[2]:.2f}")

    return mAP, ap_per_class, pr_curves

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- EDIT PATHS ----
    annotations_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\reduced_labels\2_labels\original_val_superduperreduced.csv")
    dataset_root = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\images")
    checkpoint_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments_after_vacations\sgd_lr0.01_bs4_cosine_3classes_3warmup_7constant_cosine_160426_nonstop\checkpoints\best_match_precision.pth")

    # ---- LOAD DATA ----
    annotations = pd.read_csv(annotations_path)

    dataset = EchoDataset(
        annotations_data=annotations,
        dataset_root=dataset_root,
        transforms=None
    )

    dataloader = build_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    # ---- LOAD MODEL ----
    model = build_faster_rcnn_model(num_classes=3)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    # ---- RUN mAP ----
    mAP50, ap50, curves50 = evaluate_map(
        model, dataloader, device, 3, iou_threshold=0.75
    )

    print(f"\nmAP@0.5: {mAP50:.4f}")

    # ---- PLOT CURVES ----
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,6))

    for c in curves50:
        precision, recall = curves50[c]
        plt.plot(recall, precision, label=f"Class {c}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve @ IoU=0.5")
    plt.legend()
    plt.grid(True)
    plt.show()