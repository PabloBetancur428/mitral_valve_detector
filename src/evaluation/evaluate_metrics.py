import torch
from tqdm import tqdm

from src.evaluation.metrics import (
    calculate_match_precision,
    calculate_label_precision
)


def evaluate_metrics(model, dataloader, device, iou_threshold=0.5, max_batches=None):

    model.eval()

    match_precisions = []
    label_precisions = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Metrics")):

            # Optional: limit batches for training metrics (speed)
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = [img.to(device) for img in images]

            outputs = model(images)

            for i in range(len(images)):

                pred_boxes  = outputs[i]["boxes"].cpu()
                pred_labels = outputs[i]["labels"].cpu()
                gt_boxes    = targets[i]["boxes"].cpu()
                gt_labels   = targets[i]["labels"].cpu()

                # PSAX_MV has duplicated box — deduplicate gt before evaluating
                view = targets[i]["view"]
                if view == "PSAX_MV":
                    gt_boxes  = gt_boxes[:1]    # only one real box
                    gt_labels = gt_labels[:1]

                # top-k predictions: 1 for PSAX_MV, 2 for others
                k = 1 if view == "PSAX_MV" else 2
                if len(outputs[i]["scores"]) > 0:
                    scores     = outputs[i]["scores"].cpu()
                    sorted_idx = torch.argsort(scores, descending=True)
                    pred_boxes  = pred_boxes[sorted_idx][:k]
                    pred_labels = pred_labels[sorted_idx][:k]

                # --- METRICS ---
                mp = calculate_match_precision(pred_boxes, gt_boxes, iou_threshold)
                lp = calculate_label_precision(pred_labels, gt_labels)

                match_precisions.append(mp)
                label_precisions.append(lp)

    avg_match_precision = sum(match_precisions) / len(match_precisions)
    avg_label_precision = sum(label_precisions) / len(label_precisions)

    return {
        "avg_match_precision": avg_match_precision,
        "avg_label_precision": avg_label_precision
    }