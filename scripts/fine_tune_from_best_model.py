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

import argparse
import json
import sys
import time

from src.datasets.echo_dataset import EchoDataset
from src.datasets.dataset_split import split_annotations_by_study
from src.training.dataloader import build_dataloader
from src.models.faster_rcnn_model import build_faster_rcnn_model
from src.training.train_engine import train_one_epoch
from src.evaluation.evaluate_detector import evaluate_detector
from src.evaluation.evaluate_metrics import evaluate_metrics
from src.datasets.transforms import get_train_transforms, get_val_transforms


def main():

    # =========================
    # ARGUMENTS 
    # =========================
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--step_size", type=int, default=10)   # for StepLR
    parser.add_argument("--gamma", type=float, default=0.1)    # for StepLR
    parser.add_argument("--run_name", type=str, default="default_run")

    args = parser.parse_args()

    # =========================
    # EXPERIMENT SETUP 
    # =========================
    
    exp_dir = Path(f"experiments_after_vacations/{args.run_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Redirect logs
    log_file = open(exp_dir / "train.log", "w")
    sys.stdout = log_file
    sys.stderr = log_file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(exp_dir / "tensorboard")

    # =========================
    # DATA LOADING
    # =========================
    annotations_path = Path(
        r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop\annotations_3k_reduced_labels.csv"
    )

    dataset_root = Path(
        r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\images"
    )

    train_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\reduced_labels\2_labels\train_7k_superduperreduced.csv")
    val_path   = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\reduced_labels\2_labels\original_val_superduperreduced.csv")
    val2_path = Path(rf"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\reduced_labels\2_labels\val_7k_superreduced.csv")

    if not train_path.exists():
        print(f"[INFO] Creating Train and Val split")
        annotations = pd.read_csv(annotations_path)
        train_annotations, val_annotations = split_annotations_by_study(annotations, 0.2, 42)
        train_annotations.to_csv(train_path, index=False)
        val_annotations.to_csv(val_path, index=False)
    else:
        print(f"[INFO] Train and Val splits already exist. Loading from disk.")
        train_annotations = pd.read_csv(train_path)
        val_annotations   = pd.read_csv(val_path)
        val2_annotations = pd.read_csv(val2_path)

    # For debugging
    #train_annotations = train_annotations.sample(frac=0.05, random_state=42).reset_index(drop=True)
    #val_annotations = val_annotations.sample(frac=0.08, random_state=42).reset_index(drop=True) 
    # print(f"[DEBUG]Train samples: {len(train_annotations)}, Val samples: {len(val_annotations)}")

    train_dataset = EchoDataset(train_annotations, dataset_root, get_train_transforms()) # dataset object
    val_dataset   = EchoDataset(val_annotations, dataset_root, get_val_transforms())
    val2_dataset  = EchoDataset(val2_annotations, dataset_root, get_val_transforms())

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val2_loader  = build_dataloader(val2_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # =========================
    # MODEL
    # =========================
    model = build_faster_rcnn_model(num_classes=3) # 2 classes + background 
    model.to(device)

    # Load best model
    checkpoint = torch.load(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments_after_vacations\sgd_lr0.01_bs4_cosine_3classes_3warmup_7constant_cosine_160426_nonstop\checkpoints\best_match_precision.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze backbone layers
    for name, param in model.named_parameters():
        if "backbone" in name: # Freeze feature extractor
            param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]

    # =========================
    # OPTIMIZER (MODIFIED)
    # =========================
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0005,
            nesterov=True
        )
    elif args.optimizer == "adamw":
        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=1e-4, # Low learning rate
            weight_decay=1e-4
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # =========================
    # LR SCHEDULER (MODIFIED)
    # =========================

    # --- Warmup + Cosine ---
    if args.scheduler == "cosine":

        warmup_epochs = 3
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_epochs
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=23 - warmup_epochs,
            eta_min=5e-5
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

    elif args.scheduler == "finetune_cosine":

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=5,        # short finetune
            eta_min=1e-6    # very low floor
        )

    # warmup + constant + cosine
    elif args.scheduler == "warmup_constant_cosine":

        warmup_epochs = 3
        hold_epochs = 7  

        # -------------------------
        # 1. Warmup
        # -------------------------
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,   # start at 1% of LR
            total_iters=warmup_epochs
        )

        # -------------------------
        # 2. Hold (constant LR)
        # -------------------------
        hold_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=hold_epochs
        )

        # -------------------------
        # 3. Cosine decay
        # -------------------------
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=30 - warmup_epochs - hold_epochs,
            eta_min=5e-5
        )

        # -------------------------
        # Combine all
        # -------------------------
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                hold_scheduler,
                cosine_scheduler
            ],
            milestones=[
                warmup_epochs,
                warmup_epochs + hold_epochs
            ]
        )

    # --- StepLR ---
    elif args.scheduler == "step":

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )

    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # =========================
    # CHECKPOINT SETUP
    # =========================
    checkpoint_dir = exp_dir / "checkpoints"
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

    total_start_time = time.time()  # (total training time)
    epochs = 5 # short fine tunning
    for epoch in range(epochs):

        print(f"\n===== Epoch {epoch} =====\n")

        # --- TRAIN ---
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer, debug_timing=False
        )

        # --- VALIDATION LOSS ---
        val_metrics = evaluate_detector(model, val_loader, device)
        val2_metrics = evaluate_detector(model, val2_loader, device)

        # --- METRICS ---
        val_metric_results = evaluate_metrics(model, val_loader, device)
        val2_metric_results = evaluate_metrics(model, val2_loader, device)

        train_metric_results = evaluate_metrics(
            model, train_loader, device, max_batches=10
        )

        # =========================
        # LOGGING
        # =========================
        writer.add_scalars("loss", {
            "train": train_metrics["loss"],                  # training loss
            "val": val_metrics["total_loss"],                # original val loss
            "val2": val2_metrics["total_loss"]               # NEW second val loss
        }, epoch)

        writer.add_scalars("match_precision", {
            "train": train_metric_results["avg_match_precision"],   # train MP
            "val": val_metric_results["avg_match_precision"],       # val MP
            "val2": val2_metric_results["avg_match_precision"]      # NEW val2 MP
        }, epoch)

        writer.add_scalars("label_precision", {
            "train": train_metric_results["avg_label_precision"],   # train LP
            "val": val_metric_results["avg_label_precision"],       # val LP
            "val2": val2_metric_results["avg_label_precision"]      # NEW val2 LP
        }, epoch)

        # =========================
        # MULTI-CRITERIA MODEL SELECTION
        # =========================
        current_match = val_metric_results["avg_match_precision"]
        current_label = val_metric_results["avg_label_precision"]
        current_train_loss = train_metrics["loss"]
        current_val_loss = val_metrics["total_loss"]

        print(f"[INFO] Current Metrics at epoch {epoch}:")
        print(f"       Train Loss: {current_train_loss:.4f}")
        print(f"       Val Loss: {current_val_loss:.4f}")
        print(f"       Val2 Loss: {val2_metrics['total_loss']:.4f}")           # val2 loss
        print(f"       Match Precision: {current_match:.4f}")
        print(f"       Label Precision: {current_label:.4f}")
        print(f"       Val2 Match Precision: {val2_metric_results['avg_match_precision']:.4f}")  # val2 MP
        print(f"       Val2 Label Precision: {val2_metric_results['avg_label_precision']:.4f}")  # val2 LP

        # =========================
        # INDEPENDENT MODEL SELECTION
        # =========================

        val_improvement = False

        # --- CONDITION 1: MATCH PRECISION ---
        if current_match > best_match_precision:
            best_match_precision = current_match
            #torch.save(model.state_dict(), best_match_path)

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
            }, best_match_path)

            print(f"[INFO] Saved BEST MATCH PRECISION model at epoch {epoch}")
            print(f"       MP: {best_match_precision:.4f}")

            writer.add_scalar("best/match_precision", best_match_precision, epoch)
            val_improvement = True

        # --- CONDITION 2: LABEL PRECISION ---
        if current_label > best_label_precision:
            best_label_precision = current_label

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
            val_improvement = True

        # --- CONDITION 3: TRAIN LOSS ---
        if current_train_loss < best_train_loss:
            best_train_loss = current_train_loss

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

        if current_val_loss < bes_val_loss:
            bes_val_loss = current_val_loss

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
            val_improvement = True

        # --- PATIENCE LOGIC (UNCHANGED SEMANTICS) ---
        if val_improvement:
            patience_counter = 0
        else:
            patience_counter += 0
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

    # =========================
    # TOTAL TRAINING TIME 
    # =========================
    total_time = time.time() - total_start_time
    print(f"[TIME] Total training time: {total_time:.2f}s")

    # =========================
    # FINAL SUMMARY (UPDATED)
    # =========================
    summary = {
        "best_match_precision": best_match_precision,
        "best_label_precision": best_label_precision,
        "best_train_loss": best_train_loss,
        "best_val_loss": bes_val_loss,

        # PARAMS
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "scheduler": args.scheduler,
        "step_size": args.step_size,
        "gamma": args.gamma,

        # TIME
        "total_training_time_sec": total_time
    }

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("[INFO] Training completed. Summary saved.")


if __name__ == "__main__":
    main()