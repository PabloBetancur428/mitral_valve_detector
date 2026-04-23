import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.datasets.echo_dataset import EchoDataset
from src.training.dataloader import build_dataloader
from src.models.faster_rcnn_model import build_faster_rcnn_model

from evaluation.utils_detection import match_with_fp_fn


# -----------------------
# CONFIG
# -----------------------
NUM_CLASSES = 9
NUM_CLASSES_AGGREGATED = 5
IOU_THRESHOLD = 0.5

# Map class indices → names (consistent with your dataset)
CLASS_NAMES = {
    1: "A2C_left",
    2: "A2C_right",
    3: "A3C_left",
    4: "A3C_right",
    5: "A4C_left",
    6: "A4C_right",
    7: "PSAX_MV",
    8: "PLAX_bottom",
    9: "PLAX_top",
    0: "background"  # special class
}

CLASS_NAMES_AGGREGATED = {
    1: "LEFT_MV",
    2: "RIGHT_MV",
    3: "PSAX_MV",
    4: "PLAX_bottom",
    5: "PLAX_top",
    0: "background"
} 
# -----------------------
# LABEL AGGREGATION
# -----------------------
def map_to_aggregated_class(label):
    """
    Map original class labels to aggregated anatomical classes.

    Parameters
    ----------
    label : int
        Original class label

    Returns
    -------
    int
        Aggregated class label
    """

    # Background stays background
    if label == 0:
        return 0  # background

    # LEFT MV → A2C_left, A3C_left, A4C_left
    if label in [1, 3, 5]:
        return 1  # LEFT_MV

    # RIGHT MV → A2C_right, A3C_right, A4C_right
    if label in [2, 4, 6]:
        return 2  # RIGHT_MV

    # PSAX_MV
    if label == 7:
        return 3  # PSAX_MV

    # PLAX_bottom
    if label == 8:
        return 4  # PLAX_bottom

    # PLAX_top
    if label == 9:
        return 5  # PLAX_top

    # Safety fallback
    return 0  # treat unknown as background

def compute_per_class_metrics(cm, num_classes):
    """
    Compute per-class TP, FP, FN, TN and derived metrics.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (including background at index 0)

    num_classes : int
        Total number of classes including background

    Returns
    -------
    metrics : dict
        Dictionary with metrics per class
    """


    total = np.sum(cm)
    metrics = {}

    for i in range(num_classes):

        TP = cm[i, i]  # correctly predicted class i

        FN = np.sum(cm[i, :]) - TP  # GT=i but predicted something else

        FP = np.sum(cm[:, i]) - TP  # predicted i but GT is something else

        TN = total - (TP + FN + FP)  # everything else

        recall = TP / (TP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        specificity = TN / (TN + FP + 1e-6)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        metrics[i] = {
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "f1": f1
        }

    return metrics


def compute_macro_metrics(metrics, num_classes):
    """
    Compute macro-averaged metrics (excluding background).

    Parameters
    ----------
    metrics : dict
        Output from compute_per_class_metrics

    num_classes : int
        Total number of classes including background

    Returns
    -------
    dict
        Macro metrics
    """

    import numpy as np

    recalls = []
    precisions = []
    specificities = []
    f1_scores = []

    # skip background (class 0)
    for i in range(1, num_classes):

        recalls.append(metrics[i]["recall"])
        precisions.append(metrics[i]["precision"])
        specificities.append(metrics[i]["specificity"])
        f1_scores.append(metrics[i]["f1"])

    return {
        "recall": np.mean(recalls),
        "precision": np.mean(precisions),
        "specificity": np.mean(specificities),
        "f1": np.mean(f1_scores)
    }

def plot_per_class_metrics(metrics, class_names, num_classes):
    """
    Plot per-class Recall, Precision, F1.

    Parameters
    ----------
    metrics : dict
        Output from compute_per_class_metrics

    class_names : dict
        Mapping class index → name

    num_classes : int
        Total number of classes including background
    """

    # exclude background
    print()
    names = [class_names[i] for i in range(1, num_classes)]

    recalls = [metrics[i]["recall"] for i in range(1, num_classes)]
    precisions = [metrics[i]["precision"] for i in range(1, num_classes)]
    f1_scores = [metrics[i]["f1"] for i in range(1, num_classes)]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, recalls, width, label='Recall')
    plt.bar(x, precisions, width, label='Precision')
    plt.bar(x + width, f1_scores, width, label='F1')

    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Per-Class Metrics")

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_global_metrics(macro_metrics, title):
    """
    Plot global macro metrics.

    Parameters
    ----------
    macro_metrics : dict
        Output from compute_macro_metrics
    """

    import matplotlib.pyplot as plt

    names = list(macro_metrics.keys())
    values = list(macro_metrics.values())

    plt.figure(figsize=(6, 4))

    plt.bar(names, values)

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Global Metrics (Macro) {title}")

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def main():

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # -------- PATHS (EDIT) --------
    annotations_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\val_3_5_split_prop.csv")
    dataset_root = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\images")
    checkpoint_path = Path(rf"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments\model_7k_steplr_2403_augmentations_all_data\best_model_7k_steplr_2403_augmentations__all_data.pth")
    

    # -------- LOAD DATA --------
    annotations = pd.read_csv(annotations_path)
    #annotations = annotations.sample(30).reset_index(drop=True)
    print(f"Len annotations: {len(annotations)}")



    dataset = EchoDataset(
        annotations_data=annotations,
        dataset_root=dataset_root,
        transforms=None  # IMPORTANT → no augmentation in evaluation
    )

    dataloader = build_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1
    )

    img, target = dataset[0]

    # -------- LOAD MODEL --------
    model = build_faster_rcnn_model(num_classes=10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_gt = []
    all_pred = []

    all_gt_aggregated = []
    all_pred_aggregaed = []

    # For detection metrics only
    total_gt_boxes = 0        # total GT objects
    total_pred_boxes = 0      # total predicted boxes
    total_matched = 0         # matched pairs (TP)

    # -------- INFERENCE --------
    with torch.no_grad():

        for images, targets in tqdm(dataloader):

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):

                pred_boxes = outputs[i]["boxes"].cpu()
                pred_labels = outputs[i]["labels"].cpu()
                pred_scores = outputs[i]["scores"].cpu()

                gt_boxes = targets[i]["boxes"].cpu()
                gt_labels = targets[i]["labels"].cpu()

                view = targets[i]["view"]

                # --- SORT BY SCORE ---
                sorted_idx = torch.argsort(pred_scores, descending=True)

                pred_boxes = pred_boxes[sorted_idx]
                pred_labels = pred_labels[sorted_idx]
                pred_scores = pred_scores[sorted_idx]

                # --- VIEW LOGIC ---
                if view == "PSAX_MV":

                    gt_boxes = gt_boxes[:1]
                    gt_labels = gt_labels[:1]

                    pred_boxes = pred_boxes[:1]
                    pred_labels = pred_labels[:1]

                else:

                    gt_boxes = gt_boxes[:2]
                    gt_labels = gt_labels[:2]

                    pred_boxes = pred_boxes[:2]
                    pred_labels = pred_labels[:2]

                # -------- MATCH WITH GT --------
                matched_gt, matched_pred = match_with_fp_fn(
                    pred_boxes,
                    pred_labels,
                    gt_boxes,
                    gt_labels,
                    iou_threshold=IOU_THRESHOLD
                )

                all_gt.extend(matched_gt)
                all_pred.extend(matched_pred)

                # Count GT and Pred to compute detection based metrics only
                # --- COUNT GT / PRED ---
                total_gt_boxes += len(gt_boxes)         # total GT objects
                total_pred_boxes += len(pred_boxes)     # total predictions

                # --- COUNT MATCHES (ignore label correctness) ---
                for gt, pred in zip(matched_gt, matched_pred):
                    if gt != 0 and pred != 0:
                        total_matched += 1  # TP (box matched)


                mapped_gt = [
                    map_to_aggregated_class(label) for label in matched_gt # Convert original label to matched label
                ]

                mapped_pred = [
                    map_to_aggregated_class(label) for label in matched_pred
                ]

                all_gt_aggregated.extend(mapped_gt)
                all_pred_aggregaed.extend(mapped_pred)

    # -------- CONFUSION MATRIX --------
    labels = list(range(0, NUM_CLASSES + 1))  # include background (0)
    labels_aggregaed = list(range(0, NUM_CLASSES_AGGREGATED + 1))

    cm = confusion_matrix(all_gt, all_pred, labels=labels)
    cm_agg = confusion_matrix(all_gt_aggregated, all_pred_aggregaed, labels=labels_aggregaed)
    

    # Add metrics
    total = np.sum(cm)
    num_classes = cm.shape[0]
    num_classes_agg = cm_agg.shape[0]
    
    # Compute metrics
    metrics = compute_per_class_metrics(cm, num_classes)
    metrics_agg = compute_per_class_metrics(cm_agg, num_classes_agg)

    # Compute global
    macro_metrics = compute_macro_metrics(metrics, num_classes)
    macro_metrics_agg = compute_macro_metrics(metrics_agg, num_classes_agg)

    # Plot
    plot_per_class_metrics(metrics, CLASS_NAMES, num_classes)
    plot_per_class_metrics(metrics_agg, CLASS_NAMES_AGGREGATED, num_classes_agg)

    plot_global_metrics(macro_metrics, title="Granular")
    plot_global_metrics(macro_metrics_agg, title= "Aggregated")


    # --- DETECTION METRICS ---
    match_precision = total_matched / (total_pred_boxes + 1e-6)
    match_recall = total_matched / (total_gt_boxes + 1e-6)

    match_f1 = 2 * (match_precision * match_recall) / (match_precision + match_recall + 1e-6)

    print("\n=== Detection Metrics (Class-Agnostic) ===")
    print(f"Match Precision: {match_precision:.4f}")
    print(f"Match Recall:    {match_recall:.4f}")
    print(f"Match F1:        {match_f1:.4f}")

    # Normalize (row-wise)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    cm_norm_agg = cm_agg.astype(float) / (cm_agg.sum(axis=1, keepdims=True) + 1e-6)

    # -------- PLOT --------
    # For convention with Andrea -> ground truth en el eje x y la predicción en el eje y
    cm_norm = cm_norm.T
    cm_norm_agg = cm_norm_agg.T

    class_names = [CLASS_NAMES[i] for i in labels] # add the name of the labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title("Confusion Matrix (Top-K Evaluation)")

    plt.tight_layout()
    plt.show()

    # Aggregated metrics
    class_names_agg = [CLASS_NAMES_AGGREGATED[i] for i in labels_aggregaed]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm_agg,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names_agg,
        yticklabels=class_names_agg
    )

    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title("Confusion Matrix Aggregated (Top-K Evaluation)")

    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    main()