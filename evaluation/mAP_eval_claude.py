import torch
from torchvision.ops import box_iou
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.datasets.echo_dataset import EchoDataset
from src.training.dataloader import build_dataloader
from src.models.faster_rcnn_model import build_faster_rcnn_model


# ---------------------------------------------------------------------------
# Class/View metadata — edit to match your label map
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    1: "Hinge",   # small, point-like — IoU-sensitive
    2: "MV Full (PSAX_MV)",  # large, area-based — IoU-tolerant
}

# Per-class IoU thresholds that make geometric sense for each structure.
# Hinge points are tiny: 0.5 is already aggressive. PSAX_MV is large: 0.5 is fine.
# You can override these at call time too.
DEFAULT_IOU_PER_CLASS = {
    1: 0.20,   # hinge — tighter than 0.5 will collapse AP artifically
    2: 0.50,   # full valve
}

# For the "single mAP" number we sweep these thresholds and average (COCO-style).
# If you want a simpler fixed-threshold mAP, just pass iou_thresholds=[0.5].
COCO_IOU_THRESHOLDS = [0.2, 0.3, 0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def match_detections(pred_boxes: torch.Tensor,
                     pred_scores: torch.Tensor,
                     gt_boxes: torch.Tensor,
                     iou_threshold: float):
    """
    Greedy 1-to-1 matching for ONE class on ONE image.

    NOTE: We do NOT sort here. Sorting belongs at the global accumulation
    step in evaluate_map so that confidence ordering is correct across the
    full dataset, not per-image.

    Returns:
        tp     (List[int])   — 1 if detection is a true positive
        fp     (List[int])   — 1 if detection is a false positive
        scores (List[float]) — raw confidence scores (same order as preds)
    """
    tp, fp = [], []

    matched_gt: set = set()

    for i in range(len(pred_boxes)):
        if len(gt_boxes) == 0:
            tp.append(0)
            fp.append(1)
            continue

        ious = box_iou(pred_boxes[i].unsqueeze(0), gt_boxes).squeeze(0)
        max_iou, gt_idx = torch.max(ious, dim=0)

        if max_iou.item() >= iou_threshold and gt_idx.item() not in matched_gt:
            tp.append(1)
            fp.append(0)
            matched_gt.add(gt_idx.item())
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp, pred_scores.tolist()

def box_center(box: torch.Tensor):
    """
    box: Tensor[4] = [x1, y1, x2, y2]
    returns: (cx, cy)
    """
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def center_distance(box1: torch.Tensor, box2: torch.Tensor):
    """
    Euclidean distance between box centers
    """
    c1x, c1y = box_center(box1)
    c2x, c2y = box_center(box2)
    return torch.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)

def match_detections_distance(pred_boxes: torch.Tensor,
                              pred_scores: torch.Tensor,
                              gt_boxes: torch.Tensor,
                              distance_threshold: float):
    """
    Greedy matching using center distance instead of IoU.

    A prediction is TP if:
        distance <= threshold AND GT not already matched
    """

    tp, fp = [], []
    matched_gt: set = set()

    for i in range(len(pred_boxes)):

        if len(gt_boxes) == 0:
            tp.append(0)
            fp.append(1)
            continue

        # Compute distances to all GT boxes
        distances = torch.tensor([
            center_distance(pred_boxes[i], gt_boxes[j])
            for j in range(len(gt_boxes))
        ])

        min_dist, gt_idx = torch.min(distances, dim=0)

        if min_dist.item() <= distance_threshold and gt_idx.item() not in matched_gt:
            tp.append(1)
            fp.append(0)
            matched_gt.add(gt_idx.item())
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp, pred_scores.tolist()

def compute_precision_recall(tp: np.ndarray,
                              fp: np.ndarray,
                              num_gt: int):
    """
    Cumulative precision-recall curve from globally sorted TP/FP arrays.
    Input arrays must already be sorted by descending confidence score.
    """
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall    = tp_cum / (num_gt + 1e-9)

    return precision, recall


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Area under the PR curve via the 101-point interpolation used by COCO.

    Why 101-point instead of the VOC boundary-append trick?
    The VOC method forces recall to 1.0 even when the model never reaches it,
    which artificially deflates AP for low-recall classes (your hinge points).
    The 101-point interpolation only integrates over recall levels the model
    actually achieves, giving a fairer score.
    """
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        # Precision at all recall points >= thr
        prec_at_thr = precision[recall >= thr]
        ap += (np.max(prec_at_thr) if len(prec_at_thr) > 0 else 0.0)
    return ap / 101.0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_map(model,
                 dataloader,
                 device,
                 num_classes: int,
                 iou_thresholds=None):
    """
    Evaluate mAP with separate, geometrically appropriate IoU thresholds
    per class. Also returns AP at every requested threshold so you can
    plot sensitivity curves.

    Args:
        num_classes:    Total classes including background (e.g. 3 → classes 1,2).
        iou_thresholds: List of IoU thresholds to sweep. Defaults to COCO range.

    Returns:
        results: dict with keys:
            "mAP"            → scalar, mean over all classes and thresholds
            "ap_per_class"   → {class_id: mean AP over thresholds}
            "ap_full_table"  → {iou_thr: {class_id: AP}}
            "pr_curves"      → {class_id: (precision, recall)} at class-specific IoU
    """
    if iou_thresholds is None:
        iou_thresholds = COCO_IOU_THRESHOLDS

    model.eval()

    foreground_classes = list(range(1, num_classes))   # [1, 2] for num_classes=3

    # Accumulate raw detections across the entire dataset, per class.
    # We store (score, is_tp) pairs; sorting happens once at the end.
    detections = {c: {"scores": [], "tp": [], "fp": []} for c in foreground_classes}
    num_gt_per_class = {c: 0 for c in foreground_classes}

    # We need per-class PR curves at the class-specific IoU threshold.
    # We'll run matching at each unique threshold we need.
    all_thresholds_needed = set(iou_thresholds) | set(DEFAULT_IOU_PER_CLASS.values())

    # Storage: detections_at_thr[iou_thr][class_id] = {scores, tp, fp}
    detections_at_thr = {
        thr: {c: {"scores": [], "tp": [], "fp": []} for c in foreground_classes}
        for thr in all_thresholds_needed
    }
    num_gt_at_thr = {   # GT counts don't depend on IoU, but we keep per-thr for API symmetry
        thr: {c: 0 for c in foreground_classes}
        for thr in all_thresholds_needed
    }

    # Distance-based evaluation (only once, no IoU sweep)
    DISTANCE_THRESHOLD = 10.0  # tune this

    detections_distance = {
        c: {"scores": [], "tp": [], "fp": []}
        for c in foreground_classes
    }
    num_gt_distance = {c: 0 for c in foreground_classes}

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                pred_boxes  = outputs[i]["boxes"].cpu()
                pred_labels = outputs[i]["labels"].cpu()
                pred_scores = outputs[i]["scores"].cpu()

                gt_boxes  = targets[i]["boxes"].cpu()
                gt_labels = targets[i]["labels"].cpu()
                view      = targets[i]["view"]

                # ---- GT truncation (preserving your original fix) ----
                if view == "PSAX_MV":
                    gt_boxes  = gt_boxes[:1]
                    gt_labels = gt_labels[:1]
                else:
                    gt_boxes  = gt_boxes[:2]
                    gt_labels = gt_labels[:2]

                # ---- Match per class, per threshold ----
                for thr in all_thresholds_needed:
                    for c in foreground_classes:
                        pred_mask = pred_labels == c
                        gt_mask   = gt_labels   == c

                        c_pred_boxes  = pred_boxes[pred_mask]
                        c_pred_scores = pred_scores[pred_mask]
                        c_gt_boxes    = gt_boxes[gt_mask]

                        num_gt_at_thr[thr][c] += len(c_gt_boxes)

                        tp, fp, scores = match_detections(
                            c_pred_boxes, c_pred_scores, c_gt_boxes, thr
                        )

                        detections_at_thr[thr][c]["scores"].extend(scores)
                        detections_at_thr[thr][c]["tp"].extend(tp)
                        detections_at_thr[thr][c]["fp"].extend(fp)

                for c in foreground_classes:
                    pred_mask = pred_labels == c
                    gt_mask   = gt_labels   == c

                    c_pred_boxes  = pred_boxes[pred_mask]
                    c_pred_scores = pred_scores[pred_mask]
                    c_gt_boxes    = gt_boxes[gt_mask]

                    num_gt_distance[c] += len(c_gt_boxes)

                    tp_d, fp_d, scores_d = match_detections_distance(
                        c_pred_boxes,
                        c_pred_scores,
                        c_gt_boxes,
                        DISTANCE_THRESHOLD
                    )

                    detections_distance[c]["scores"].extend(scores_d)
                    detections_distance[c]["tp"].extend(tp_d)
                    detections_distance[c]["fp"].extend(fp_d)
    # ---------------------------------------------------------------------------
    # Compute AP for every (class, threshold) combination
    # ap_full_table[thr][class_id] = AP value
    # ---------------------------------------------------------------------------
    ap_full_table = {}
    pr_curves     = {}   # only at class-specific threshold

    for thr in all_thresholds_needed:
        ap_full_table[thr] = {}

        for c in foreground_classes:
            scores = np.array(detections_at_thr[thr][c]["scores"])
            tp     = np.array(detections_at_thr[thr][c]["tp"])
            fp     = np.array(detections_at_thr[thr][c]["fp"])
            n_gt   = num_gt_at_thr[thr][c]

            if n_gt == 0 or len(scores) == 0:
                ap_full_table[thr][c] = 0.0
                if thr == DEFAULT_IOU_PER_CLASS.get(c, iou_thresholds[0]):
                    pr_curves[c] = (np.array([0.0]), np.array([0.0]))
                continue

            # Sort globally by descending confidence — this is the correct place
            sorted_idx = np.argsort(-scores)
            tp = tp[sorted_idx]
            fp = fp[sorted_idx]

            precision, recall = compute_precision_recall(tp, fp, n_gt)
            ap = compute_ap(precision, recall)

            ap_full_table[thr][c] = ap

            # Store PR curve at this class's designated threshold
            if thr == DEFAULT_IOU_PER_CLASS.get(c, iou_thresholds[0]):
                pr_curves[c] = (precision, recall)
    ap_distance = {}

    for c in foreground_classes:
        scores = np.array(detections_distance[c]["scores"])
        tp     = np.array(detections_distance[c]["tp"])
        fp     = np.array(detections_distance[c]["fp"])
        n_gt   = num_gt_distance[c]

        if n_gt == 0 or len(scores) == 0:
            ap_distance[c] = 0.0
            continue

        sorted_idx = np.argsort(-scores)
        tp = tp[sorted_idx]
        fp = fp[sorted_idx]

        precision, recall = compute_precision_recall(tp, fp, n_gt)
        ap = compute_ap(precision, recall)

        ap_distance[c] = ap
    # ---------------------------------------------------------------------------
    # Summary metrics
    # ---------------------------------------------------------------------------

    # AP per class averaged over the requested sweep thresholds
    ap_per_class = {}
    for c in foreground_classes:
        ap_per_class[c] = float(np.mean([
            ap_full_table[thr][c] for thr in iou_thresholds
        ]))

    # Overall mAP = mean over classes and thresholds
    mAP = float(np.mean(list(ap_per_class.values())))

    return {
        "mAP":          mAP,
        "ap_per_class": ap_per_class,
        "ap_full_table": ap_full_table,
        "pr_curves":    pr_curves,
        "ap_distance":  ap_distance,   # <-- add this
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_results(results: dict, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = COCO_IOU_THRESHOLDS

    print("\n" + "=" * 60)
    print("  mAP EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n  mAP (mean over classes & IoU thresholds): {results['mAP']:.4f}")

    print("\n  AP per class (averaged over IoU thresholds):")
    for c, ap in results["ap_per_class"].items():
        name = CLASS_NAMES.get(c, f"Class {c}")
        print(f"    [{c}] {name:<30s}  AP = {ap:.4f}")

    print("\n  AP per class × IoU threshold breakdown:")
    header = f"  {'Class':<30s}" + "".join(f"  IoU={t:.2f}" for t in iou_thresholds)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in results["ap_per_class"]:
        name = CLASS_NAMES.get(c, f"Class {c}")
        row = f"  [{c}] {name:<28s}"
        for t in iou_thresholds:
            ap_val = results["ap_full_table"].get(t, {}).get(c, float("nan"))
            row += f"    {ap_val:.4f}"
        print(row)

    print("=" * 60 + "\n")

    if "ap_distance" in results:
        print("\n  Distance-based AP (center distance):")
        for c, ap in results["ap_distance"].items():
            name = CLASS_NAMES.get(c, f"Class {c}")
            print(f"    [{c}] {name:<30s}  AP_dist = {ap:.4f}")


def plot_pr_curves(results: dict, save_path: str = None):
    fig, axes = plt.subplots(1, len(results["pr_curves"]), figsize=(6 * len(results["pr_curves"]), 5))

    if len(results["pr_curves"]) == 1:
        axes = [axes]

    for ax, (c, (precision, recall)) in zip(axes, results["pr_curves"].items()):
        name = CLASS_NAMES.get(c, f"Class {c}")
        iou_thr = DEFAULT_IOU_PER_CLASS.get(c, 0.5)
        ap = results["ap_per_class"].get(c, 0.0)

        ax.plot(recall, precision, color="steelblue", linewidth=2)
        ax.fill_between(recall, precision, alpha=0.15, color="steelblue")
        ax.set_xlabel("Recall",    fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"{name}\nAP={ap:.4f}  @IoU={iou_thr}", fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Precision-Recall Curves  |  mAP = {results['mAP']:.4f}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"PR curves saved to: {save_path}")

    plt.show()


def plot_iou_sensitivity(results: dict, iou_thresholds=None):
    """
    Shows how AP drops as IoU threshold increases — makes the geometric
    asymmetry between small (hinge) and large (PSAX_MV) structures visible.
    """
    if iou_thresholds is None:
        iou_thresholds = COCO_IOU_THRESHOLDS

    plt.figure(figsize=(7, 5))

    for c in results["ap_per_class"]:
        name = CLASS_NAMES.get(c, f"Class {c}")
        aps  = [results["ap_full_table"].get(t, {}).get(c, 0.0) for t in iou_thresholds]
        plt.plot(iou_thresholds, aps, marker="o", label=name, linewidth=2)

    plt.xlabel("IoU Threshold", fontsize=12)
    plt.ylabel("AP", fontsize=12)
    plt.title("AP vs. IoU Threshold (IoU Sensitivity)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- EDIT PATHS ----
    annotations_path = Path(
        r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector"
        r"\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\reduced_labels"
        r"\2_labels\original_val_superduperreduced.csv"
    )
    dataset_root = Path(
        r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector"
        r"\data\processed_cone_dataset_prop_aSaco\images"
    )
    checkpoint_path = Path(
        rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments_after_vacations\3classes_3warmup_15constant_cosine_200426_full\checkpoints\best_match_precision.pth"
    )

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
    model = build_faster_rcnn_model(num_classes=3)   # bg + class1 + class2
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ---- RUN EVALUATION ----
    # Sweep COCO IoU thresholds [0.50 … 0.75] for the aggregate mAP.
    # PR curves are plotted at each class's geometrically appropriate threshold
    # (see DEFAULT_IOU_PER_CLASS at the top of the file).
    results = evaluate_map(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=3,
        iou_thresholds=COCO_IOU_THRESHOLDS,
    )

    # ---- PRINT FULL REPORT ----
    print_results(results, iou_thresholds=COCO_IOU_THRESHOLDS)

    # ---- PLOTS ----
    plot_pr_curves(results, save_path="pr_curves.png")
    plot_iou_sensitivity(results, iou_thresholds=COCO_IOU_THRESHOLDS)