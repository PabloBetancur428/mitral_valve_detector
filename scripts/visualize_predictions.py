import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pandas as pd
import argparse
import random
from src.datasets.echo_dataset import EchoDataset
from src.datasets.dataset_split import split_annotations_by_study
from src.training.dataloader import build_dataloader
from src.models.faster_rcnn_model import build_faster_rcnn_model
from torchvision.ops import box_iou



CLASS_MAP = {
    1: "LEFT_MV",
    2: "RIGHT_MV",
    3: "PSAX_MV",
    4: "PLAX_bottom",
    5: "PLAX_top",
    0: "background"
} 

CLASS_COLORS = {
    1: "red",
    2: "blue",
    3: "magenta",
    4: "green",
    5: "purple",
    0: "gray"
}


# -----------------------------
# VISUALIZATION FUNCTION
# -----------------------------
def draw_gt_vs_pred(
    image,
    gt_boxes,
    gt_labels,
    pred_boxes,
    pred_labels=None,
    scores=None
):
    """
    Improved visualization:
    - Indexed predictions (number on box)
    - Clean side panel aligned with indices
    - Better spacing and readability
    """


    # -------------------------
    # SORT PREDICTIONS BY SCORE
    # -------------------------
    if scores is not None and len(scores) > 0:
        sorted_idx = scores.argsort(descending=True)
        pred_boxes = pred_boxes[sorted_idx]
        pred_labels = pred_labels[sorted_idx]
        scores = scores[sorted_idx]

    # -------------------------
    # IMAGE PREP (CHW → HWC)
    # -------------------------
    img = image.permute(1, 2, 0).cpu().numpy()

    # -------------------------
    # FIGURE LAYOUT (3 columns)
    # -------------------------
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.7])

    ax_gt = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1])
    ax_panel = fig.add_subplot(gs[2])

    # -------------------------
    # GROUND TRUTH
    # -------------------------
    ax_gt.imshow(img)
    ax_gt.set_title("Ground Truth", fontsize=12, weight="bold")

    for i, box in enumerate(gt_boxes):
        label = int(gt_labels[i].item())
        color = CLASS_COLORS.get(label, "white")

        x1, y1, x2, y2 = box

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax_gt.add_patch(rect)

        ax_gt.text(
            x1,
            y1 - 5,
            f"{label}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor=color, alpha=0.8, pad=1)
        )

    ax_gt.axis("off")

    # -------------------------
    # PREDICTIONS
    # -------------------------
    ax_pred.imshow(img)
    ax_pred.set_title("Predictions", fontsize=12, weight="bold")

    pred_text_lines = []

    for i, box in enumerate(pred_boxes):
        label = int(pred_labels[i].item())
        score = float(scores[i].item()) if scores is not None else 0.0
        color = CLASS_COLORS.get(label, "white")

        x1, y1, x2, y2 = box

        # BOX
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax_pred.add_patch(rect)

        # INDEX LABEL ON BOX (key improvement)
        ax_pred.text(
            x1,
            y1 - 5,
            f"{i}",
            color="white",
            fontsize=9,
            weight="bold",
            bbox=dict(facecolor=color, alpha=0.9, pad=2)
        )

        class_name = CLASS_MAP.get(label, "unknown")

        pred_text_lines.append({
            "idx": i,
            "label": label,
            "name": class_name,
            "score": score,
            "color": color
        })

    ax_pred.axis("off")

    # -------------------------
    # SIDE PANEL (WITH BACKGROUND)
    # -------------------------
    ax_panel.set_title("Detections", fontsize=12, weight="bold")

    # Dark background (key improvement)
    ax_panel.set_facecolor("#1c899716")  # soft dark gray

    # Remove borders/ticks
    for spine in ax_panel.spines.values():
        spine.set_visible(False)

    ax_panel.set_xticks([])
    ax_panel.set_yticks([])

    y_start = 0.95
    line_h = 0.07

    for i, item in enumerate(pred_text_lines):
        y = y_start - i * line_h

        text = f"[{item['idx']}] {item['name']}  ({item['score']:.2f})"

        ax_panel.text(
            0.05,
            y,
            text,
            color=item["color"],
            fontsize=10,
            verticalalignment="top",
            transform=ax_panel.transAxes,
            fontweight="medium"
        )

    # -------------------------
    # LEGEND
    # -------------------------
    legend_patches = []

    for class_id, name in CLASS_MAP.items():
        color = CLASS_COLORS[class_id]

        patch = patches.Patch(
            color=color,
            label=f"{class_id}: {name}"
        )
        legend_patches.append(patch)

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(CLASS_MAP),
        fontsize=8
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def main(view = None, num_samples = 10):

    print("--------------------------- [INFO] Checking final model---------------------------")
    annotations_path = Path(
        rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\val_3_5_split_reduced_labels.csv"
    )

    dataset_root = Path(
        rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\images"
    )

    val_ann = pd.read_csv(annotations_path)
    
    # Needed if you gonna use an entire dataframe
    #train_ann, val_ann = split_annotations_by_study(annotations, val_ratio=0.2, random_state=42)

    # Filter per view if required
    if view is not None:
        val_ann = val_ann[val_ann['view'] == view]
        print(f"Len dataset after filtering by view {view}: {len(val_ann)}")

    dataset = EchoDataset(val_ann.reset_index(drop=True), 
                          dataset_root,
                          transforms=None)

    print(f"Validation dataset: {len(dataset)}")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"
    model = build_faster_rcnn_model(num_classes=6)

    model.to(device)

    model.eval()

    model.load_state_dict(
        torch.load(r"experiments\best_models_new_augs_7k_3103_highLR\best_match_precision.pth")
    )

    print("[INFO] Model loaded")
    # Predict X random samples
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
    
        image, target = dataset[idx]
        image_device = image.to(device)

        with torch.no_grad():
            predictions = model([image_device])

        pred_boxes = predictions[0]["boxes"].cpu()
        pred_labels = predictions[0]["labels"].cpu()
        scores = predictions[0]["scores"].cpu()

        keep = scores > 0.2
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        scores = scores[keep]

        # Print predictions info
        print(f"\nSample {idx}")
        print(f"GT boxes: {len(target['boxes'])}")
        print(f"Pred boxes: {len(pred_boxes)}")

        for i in range(len(pred_boxes)):
            print(
                f"Pred {i}: "
                f"class={pred_labels[i].item()} "
                f"score={scores[i]:.3f}"
            )

        # Plot predictions
        gt_boxes = target["boxes"]

        if view == "PSAX_MV":
            iou = box_iou(pred_boxes, gt_boxes)
            print("IoU matrix: \n ", iou)

        # Draw gt and predicted
        draw_gt_vs_pred(
            image,
            target["boxes"],
            target["labels"], 
            pred_boxes,
            pred_labels,
            scores
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--view",
        type=str.upper,
        default=None,
        help="Filter predictions by view (A2C, A3C, A4C, PLAX, PSAX_MV)"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to make inference on"
    )
    
    args = parser.parse_args()

    main(view=args.view, num_samples=args.num_samples or 5)

