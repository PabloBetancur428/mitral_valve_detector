import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from torchvision.ops import box_iou

from src.datasets.echo_dataset import EchoDataset
from src.training.dataloader import build_dataloader
from src.models.faster_rcnn_model import build_faster_rcnn_model


# -----------------------
# CONFIG
# -----------------------

NUM_CLASSES = 6  # includes background (0)

CLASS_NAMES = {
    1: "LEFT_MV",
    2: "RIGHT_MV",
    3: "PSAX_MV",
    4: "PLAX_bottom",
    5: "PLAX_top",
    0: "background"
} 


CONFIDENCE_THRESHOLD = 0.5


# -----------------------
# CORE FUNCTION
# -----------------------

def collect_iou_class_data(model, dataloader, device):
    """
    Core evaluation loop.

    What this does (IMPORTANT):
    ---------------------------
    For each GT object:
        - Finds the prediction with highest IoU
        - Stores:
            * IoU
            * GT label
            * Predicted label
            * Whether classification is correct

    This is the correct way to analyze:
        P(correct_class | IoU)

    Returns
    -------
    records : list of dict
        Each entry corresponds to ONE GT object
    """

    records = []  # This will store all observations

    model.eval()

    with torch.no_grad():

        for images, targets in tqdm(dataloader):

            # Move images to device
            images = [img.to(device) for img in images]

            # Run inference
            outputs = model(images)

            # Iterate per image in batch
            for i in range(len(images)):

                # -----------------------
                # PREDICTIONS
                # -----------------------
                pred_boxes = outputs[i]["boxes"].cpu()
                pred_labels = outputs[i]["labels"].cpu()
                pred_scores = outputs[i]["scores"].cpu()

                # Apply confidence filtering (same as your pipeline)
                keep = pred_scores > CONFIDENCE_THRESHOLD
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                # -----------------------
                # GROUND TRUTH
                # -----------------------
                gt_boxes = targets[i]["boxes"].cpu()
                gt_labels = targets[i]["labels"].cpu()

                # -----------------------
                # HANDLE EMPTY CASES
                # -----------------------
                if len(gt_boxes) == 0:
                    continue  # no GT → nothing to evaluate

                if len(pred_boxes) == 0:
                    # No predictions → all GTs are missed
                    for gt_label in gt_labels:

                        records.append({
                            "iou": 0.0,  # no overlap
                            "gt_class": int(gt_label.item()),
                            "pred_class": 0,  # background
                            "correct": 0
                        })
                    continue

                # -----------------------
                # COMPUTE IoU MATRIX
                # -----------------------
                # Shape: [num_preds, num_gt]
                iou_matrix = box_iou(pred_boxes, gt_boxes)

                # -----------------------
                # MATCH EACH GT → BEST PRED
                # -----------------------
                # We iterate over GT (not predictions)
                # because we want one record per GT object
                for gt_idx in range(len(gt_boxes)):

                    # Get IoU values between all preds and this GT
                    ious = iou_matrix[:, gt_idx]

                    # Find best prediction for this GT
                    best_pred_idx = torch.argmax(ious)
                    best_iou = ious[best_pred_idx].item()

                    # Retrieve labels
                    gt_class = int(gt_labels[gt_idx].item())
                    pred_class = int(pred_labels[best_pred_idx].item())

                    # Classification correctness
                    correct = int(gt_class == pred_class)

                    # Store record
                    records.append({
                        "iou": best_iou,
                        "gt_class": gt_class,
                        "pred_class": pred_class,
                        "correct": correct
                    })

    return records


# -----------------------
# PLOTTING FUNCTIONS
# -----------------------

def plot_iou_distribution_with_confusions(df, class_names):
    """
    For each GT class:
        - Plot IoU distribution
        - Separate:
            * Correct predictions
            * Incorrect predictions split by predicted class
    """

    classes = sorted(df["gt_class"].unique())

    for gt_cls in classes:

        if gt_cls == 0:
            continue  # skip background GT

        class_name = class_names.get(gt_cls, str(gt_cls))

        df_cls = df[df["gt_class"] == gt_cls]

        plt.figure(figsize=(9, 5))

        # -----------------------
        # CORRECT
        # -----------------------
        correct = df_cls[df_cls["correct"] == 1]["iou"]

        plt.hist(
            correct,
            bins=20,
            alpha=0.6,
            label="Correct",
            density=True
        )

        # -----------------------
        # INCORRECT SPLIT BY CLASS
        # -----------------------
        incorrect_df = df_cls[df_cls["correct"] == 0]

        pred_classes = sorted(incorrect_df["pred_class"].unique())

        for pred_cls in pred_classes:

            subset = incorrect_df[incorrect_df["pred_class"] == pred_cls]["iou"]

            pred_name = class_names.get(pred_cls, str(pred_cls))

            plt.hist(
                subset,
                bins=20,
                alpha=0.5,
                label=f"Wrong → {pred_name}",
                density=True
            )

        # -----------------------
        # PLOT
        # -----------------------
        plt.xlabel("IoU")
        plt.ylabel("Density")
        plt.title(f"IoU Distribution with Confusions - GT = {class_name}")

        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


def plot_accuracy_vs_iou(df):
    """
    More advanced plot:
        Accuracy as function of IoU

    This is the CLEANEST diagnostic.
    """

    # Bin IoU
    bins = np.linspace(0, 1, 11)

    df["iou_bin"] = pd.cut(df["iou"], bins=bins)

    grouped = df.groupby("iou_bin")["correct"].mean()

    plt.figure(figsize=(8, 5))

    grouped.plot(marker="o")

    plt.xlabel("IoU bin")
    plt.ylabel("Accuracy")
    plt.title("P(correct class | IoU)")

    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    plt.show()


# -----------------------
# MAIN
# -----------------------

def main():

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- PATHS (USE YOURS) --------
    annotations_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\original_6class_val_3_5.csv")
    dataset_root = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\images")
    checkpoint_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments_gone\sgd_lr0.01_bs4_cosine\checkpoints\best_val_loss.pth")


    # -------- LOAD DATA --------
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

    # -------- LOAD MODEL --------
    model = build_faster_rcnn_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # -------- COLLECT DATA --------
    records = collect_iou_class_data(model, dataloader, device)

    df = pd.DataFrame(records)

    print("\nSample of collected data:")
    print(df.head())

    # -------- PLOTS --------
    plot_iou_distribution_with_confusions(df, CLASS_NAMES)
    plot_accuracy_vs_iou(df)


if __name__ == "__main__":
    main()