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

NUM_CLASSES = 3  # includes background (0)

CLASS_NAMES = {
    1: "HINGE",
    2: "PSAX_MV",
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

                    # Confidence score of that prediction
                    confidence = float(pred_scores[best_pred_idx].item())

                    # Classification correctness
                    correct = int(gt_class == pred_class)

                    # Store record
                    records.append({
                        "iou": best_iou,
                        "confidence": confidence,
                        "gt_class": gt_class,
                        "pred_class": pred_class,
                        "correct": correct
                    })

    return records


# -----------------------
# PLOTTING FUNCTIONS
# -----------------------
# Confusion here-> mispredictions
def plot_confidence_distribution_with_confusions(df, class_names):
    """
    For each GT class:
        - Plot distribution of model confidence
        - Separate:
            * Correct predictions
            * Incorrect predictions split by predicted class

    This answers:
        "When the model is confident, is it correct?"
        "When it is wrong, which class is it confident about?"
    """

    # Get all GT classes present in the dataframe
    classes = sorted(df["gt_class"].unique())

    # Loop over each GT class independently
    for gt_cls in classes:

        # Skip background GT (we don't analyze it)
        if gt_cls == 0:
            continue

        # Convert class index to readable name
        class_name = class_names.get(gt_cls, str(gt_cls))

        # Filter dataframe for this GT class only
        df_cls = df[df["gt_class"] == gt_cls]

        # Create a new figure for this class
        plt.figure(figsize=(9, 5))

        # -----------------------
        # CORRECT PREDICTIONS
        # -----------------------

        # Take confidence values where prediction was correct
        correct = df_cls[df_cls["correct"] == 1]["confidence"]

        # Plot histogram of confidence for correct predictions
        plt.hist(
            correct,            # data to plot
            bins=20,            # number of bins
            alpha=0.6,          # transparency (so overlaps are visible)
            label="Correct",    # legend label
            density=False        # normalize → compare shapes, not counts
        )

        # -----------------------
        # INCORRECT PREDICTIONS
        # -----------------------

        # Filter only incorrect predictions
        incorrect_df = df_cls[df_cls["correct"] == 0]

        # Get all predicted classes that appear in errors
        pred_classes = sorted(incorrect_df["pred_class"].unique())

        # Loop over each wrong predicted class
        for pred_cls in pred_classes:

            # Take confidence values for this specific confusion
            subset = incorrect_df[
                incorrect_df["pred_class"] == pred_cls
            ]["confidence"]

            # Skip empty or NaN-only data
            if len(subset) == 0 or subset.isna().all():
                print(f"Skipping GT={class_name} → Pred={class_names.get(pred_cls, str(pred_cls))} (no data)")
                continue

            # Convert predicted class to readable name
            pred_name = class_names.get(pred_cls, str(pred_cls))

            # Plot histogram for this specific confusion
            plt.hist(
                subset,
                bins=20,
                alpha=0.6,
                label=f"Wrong → {pred_name}",
                density=False
            )

        # -----------------------
        # FINAL PLOT SETTINGS
        # -----------------------

        plt.xlabel("Confidence")  # X-axis = model confidence
        plt.ylabel("Frequency")     # normalized distribution
        plt.title(f"Confidence Distribution - GT = {class_name}")

        # Show legend with all curves
        plt.legend()

        # Light grid for readability
        plt.grid(alpha=0.3)

        # Adjust layout to avoid overlaps
        plt.tight_layout()

        # Display plot
        plt.show()


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
            density=False
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
                alpha=0.8,
                label=f"Wrong → {pred_name}",
                density=False
            )

        # -----------------------
        # PLOT
        # -----------------------
        plt.xlabel("IoU")
        plt.ylabel("Frequency")
        plt.title(f"IoU Distribution with Confusions - GT = {class_name}")

        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(0.2, 1)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- PATHS (USE YOURS) --------
    annotations_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\reduced_labels\2_labels\original_val_superduperreduced.csv")
    dataset_root = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\images")
    checkpoint_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments_after_vacations\sgd_lr0.01_bs4_cosine_3classes_3warmup_7constant_cosine_140426_augs05\checkpoints\best_match_precision.pth")


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
    print(f"\nTotal records collected: {len(df)}")
    print(df["confidence"].isna().sum())

    # -------- PLOTS --------
    plot_iou_distribution_with_confusions(df, CLASS_NAMES)
    # -----------------------
    # CONFIDENCE ANALYSIS
    # -----------------------
    plot_confidence_distribution_with_confusions(df, CLASS_NAMES)
    plot_accuracy_vs_iou(df)


if __name__ == "__main__":
    main()