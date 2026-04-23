"""
train.py

Main training script for the mitral valve detection model.
"""

from src.utils.seed import set_seed
set_seed(42)

import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import pandas as pd

from src.datasets.echo_dataset import EchoDataset
from src.datasets.dataset_split import split_annotations_by_study
from src.training.dataloader import build_dataloader
from src.models.faster_rcnn_model import build_faster_rcnn_model
from src.training.train_engine import train_one_epoch
from src.evaluation.evaluate_detector import evaluate_detector
from src.evaluation.evaluate_metrics import evaluate_metrics
from src.datasets.transforms import get_train_transforms, get_val_transforms


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter("experiments/tensorboard_logs_new_augs_3103_alldata_LowLR_batch4")

    # =========================
    # DATA LOADING
    # =========================
    annotations_path = Path(
        r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop\annotations_3k_reduced_labels.csv"
    )

    dataset_root = Path(
        r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\images"
    )

    # Checking with superreduced labels (only 3 classes: left hinge, right hinge, PSAX)
    train_path = Path(r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\reduced_labels\train_7k_superreduced.csv")
    val_path   = Path(r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\reduced_labels\val_7k_superreduced.csv")

    if not train_path.exists():
        print(f"[INFO] Creating Train and Val split")
        annotations = pd.read_csv(annotations_path)
        train_annotations, val_annotations = split_annotations_by_study(annotations, 0.2, 42)
        train_annotations.to_csv(train_path, index=False)
        val_annotations.to_csv(val_path, index=False)
    else:
        train_annotations = pd.read_csv(train_path)
        val_annotations   = pd.read_csv(val_path)

    # For debugging
    #train_annotations = train_annotations.sample(frac=0.15, random_state=42).reset_index(drop=True)
    #val_annotations = val_annotations.sample(frac=0.1, random_state=42).reset_index(drop=True) 
    #print(f"[DEBUG]Train samples: {len(train_annotations)}, Val samples: {len(val_annotations)}")

    train_dataset = EchoDataset(train_annotations, dataset_root, get_train_transforms()) # dataset object
    val_dataset   = EchoDataset(val_annotations, dataset_root, get_val_transforms())

    train_loader = build_dataloader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader   = build_dataloader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # =========================
    # MODEL
    # =========================
    model = build_faster_rcnn_model(num_classes=6)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer: AdamW
    # optimizer = torch.optim.AdamW(
    #     params,
    #     lr=0.002,              # keep same initial LR (you may tune later)
    #     weight_decay=0.0005    # decoupled weight decay (true L2 regularization)
    # )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=10,        # first restart after 10 epochs
    #     T_mult=2,      # cycle length doubles each restart
    #     eta_min=1e-6
    # )

    # optimizer = torch.optim.SGD(
    #     params,
    #     lr=0.02,              # IMPORTANT: higher LR for SGD (10x AdamW typical)
    #     momentum=0.9,
    #     weight_decay=0.0005,
    #     nesterov=True         # small but consistent gain
    # )


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    optimizer = torch.optim.SGD(
    params,
    lr=0.01,              # IMPORTANT: higher LR for SGD (10x AdamW typical)
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True         # small but consistent gain
    )

    # =========================
    # LR SCHEDULER (WARMUP + COSINE)
    # =========================

    # --- Warmup (first few epochs stabilize training) ---
    warmup_epochs = 3
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,   # Less extreme
        total_iters=warmup_epochs
    )

    # --- Main scheduler ---
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=23 - warmup_epochs,   # remaining epochs
        eta_min=5e-5
    )

    # --- Combine both ---
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # =========================
    # CHECKPOINT SETUP
    # =========================
    checkpoint_dir = Path("experiments/best_models_new_augs_7k_3103_LowLR_batch4")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_match_path = checkpoint_dir / "best_match_precision.pth"   # stores best MP model
    best_label_path = checkpoint_dir / "best_label_precision.pth"   # stores best LP model
    best_loss_path  = checkpoint_dir / "best_train_loss.pth"        # stores best loss model
    bes_val_loss_path = checkpoint_dir / "best_val_loss.pth"
    # --- TRACK BEST VALUES ---
    best_match_precision = -1.0     # maximize
    best_label_precision = -1.0     # maximize
    best_train_loss = float("inf")  # minimize
    bes_val_loss = float("inf")

    patience = 5
    patience_counter = 0

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(30):

        print(f"\n===== Epoch {epoch} =====\n")

        # --- TRAIN ---
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer, debug_timing=False
        )

        # --- VALIDATION LOSS ---
        val_metrics = evaluate_detector(model, val_loader, device)

        # --- METRICS ---
        val_metric_results = evaluate_metrics(model, val_loader, device)

        train_metric_results = evaluate_metrics(
            model, train_loader, device, max_batches=10
        )

        # =========================
        # LOGGING
        # =========================
        writer.add_scalars("loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["total_loss"]
        }, epoch)

        writer.add_scalars("match_precision", {
            "train": train_metric_results["avg_match_precision"],
            "val": val_metric_results["avg_match_precision"]
        }, epoch)

        writer.add_scalars("label_precision", {
            "train": train_metric_results["avg_label_precision"],
            "val": val_metric_results["avg_label_precision"]
        }, epoch)

        # =========================
        # MULTI-CRITERIA MODEL SELECTION
        # =========================
        current_match = val_metric_results["avg_match_precision"]
        current_label = val_metric_results["avg_label_precision"]
        current_train_loss = train_metrics["loss"]
        current_val_loss = val_metrics["total_loss"]

        print(f"[INFO] Current Metrics at epoch {epoch}:")
        print(f"       Match Precision: {current_match:.4f}")
        print(f"       Label Precision: {current_label:.4f}")
        print(f"       Train Loss: {current_train_loss:.4f}")
        print(f"       Val Loss: {current_val_loss:.4f}")
        # =========================
        # INDEPENDENT MODEL SELECTION
        # =========================

        # Track if ANY improvement happened (only for patience logic)
        val_improvement = False  # this replaces the old "improved" flag but only for early stopping


        # --- CONDITION 1: MATCH PRECISION ---
        if current_match > best_match_precision:  # check improvement (maximize)
            best_match_precision = current_match  # update best value

            # save model ONLY for match precision
            torch.save(model.state_dict(), best_match_path)

            print(f"[INFO] Saved BEST MATCH PRECISION model at epoch {epoch}")
            print(f"       MP: {best_match_precision:.4f}")

            writer.add_scalar("best/match_precision", best_match_precision, epoch)

            val_improvement = True  # mark improvement


        # --- CONDITION 2: LABEL PRECISION ---
        if current_label > best_label_precision:  # check improvement (maximize)
            best_label_precision = current_label  # update best value

            # save model ONLY for label precision
            #torch.save(model.state_dict(), best_label_path)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "metrics": {
                    "match_precision": current_match,
                    "label_precision": current_label,
                    "val_loss": current_val_loss,
                }
            }, best_label_path)

            print(f"[INFO] Saved BEST LABEL PRECISION model at epoch {epoch}")
            print(f"       LP: {best_label_precision:.4f}")

            writer.add_scalar("best/label_precision", best_label_precision, epoch)

            val_improvement = True  # mark improvement


        # --- CONDITION 3: TRAIN LOSS ---
        if current_train_loss < best_train_loss:  # check improvement (minimize)
            best_train_loss = current_train_loss  # update best value

            # save model ONLY for train loss
            #torch.save(model.state_dict(), best_loss_path)
            torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "metrics": {
                        "match_precision": current_match,
                        "label_precision": current_label,
                        "val_loss": current_val_loss,
                    }
            }, best_loss_path)

            print(f"[INFO] Saved BEST TRAIN LOSS model at epoch {epoch}")
            print(f"       Loss: {best_train_loss:.4f}")

            writer.add_scalar("best/train_loss", best_train_loss, epoch)

            ## NOTE: Do NOT update early stopping
            #any_improvement = True  # mark improvement


        if current_val_loss < bes_val_loss :  # check improvement (minimize)
            bes_val_loss = current_val_loss  # update best value

            # save model ONLY for train loss
            #torch.save(model.state_dict(), bes_val_loss_path)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "metrics": {
                    "match_precision": current_match,
                    "label_precision": current_label,
                    "val_loss": current_val_loss,
                }
            }, bes_val_loss_path)

            print(f"[INFO] Saved BEST VAL LOSS model at epoch {epoch}")
            print(f"       Loss: {bes_val_loss:.4f}")

            writer.add_scalar("best/val_loss", bes_val_loss, epoch)

            val_improvement = True  # mark improvement


        # --- PATIENCE LOGIC (UNCHANGED SEMANTICS) ---
        if val_improvement:
            patience_counter = 0  # reset if ANY metric improved
        else:
            patience_counter += 1  # increment otherwise
            print(f"[INFO] No VAL improvement in either Loss - Map - Lap: ({patience_counter}/{patience})")

        # =========================
        # EARLY STOPPING
        # =========================
        if patience_counter >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

        # =========================
        # LR UPDATE
        # =========================
        scheduler.step()
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)


if __name__ == "__main__":
    main()