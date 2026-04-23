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
from src.datasets.transforms import get_train_transforms, get_val_transforms


def main():

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    print(f"Using device: {device}")

    # Writer for tracking training
    log_dir = "experiments/tensorboard_logs_reduce_plateu_7k"
    writer = SummaryWriter(log_dir)

    ### Old way to read data before cone extraction
    # annotations_path = Path(
    #     r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\AoVdetector\data\annotations_boxes\mitral_annotations_2502.csv"
    # )

    # dataset_root = Path(
    #     r"\\NAS3_Z\all\DB_TARTAGLIA\496_estudios_tartaglia_jmcrespi"
    # )

    annotations_path = Path(
            #r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset\secure_annotations\annotations_fixed_1703.csv" # before propagatin
            r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\annotations_propagated_aSaco_fixed.csv"
    )

    dataset_root = Path(
        r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\images"
    )

    train_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\train_3_5_split_prop.csv")
    val_path = Path(rf"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop_aSaco\split_augmented_3_5\val_3_5_split_prop.csv")

    og = False
    if not train_path.exists():
        
        annotations = pd.read_csv(annotations_path)
        # Split by study to avoid dataleakage
        train_annotations, val_annotations = split_annotations_by_study(annotations, val_ratio=0.2, random_state=42)
        
        train_annotations.to_csv(train_path, index=False)
        val_annotations.to_csv(val_path, index=False)

        print(f"[INFO] Train annotations created and saved at: {train_path}")
        print(f"[INFO] Val annotations created and saved at: {val_path}")

    else:

        print(f"[INFO] Train annotations loaded from: {train_path}")
        print(f"[INFO] Val annotations loaded from: {val_path}")
        train_annotations = pd.read_csv(train_path)
        val_annotations   = pd.read_csv(val_path)

        if og:
            print(f"[INFO] Training only with OG frames - offset0")
            train_annotations = train_annotations[train_annotations["processed_image"].str.contains('offset0', na=False)].copy()
            val_annotations   = val_annotations[val_annotations["processed_image"].str.contains('offset0', na=False)].copy()

    # For debugging:
    #train_annotations = train_annotations.sample(150).reset_index(drop=True)
    
    #val_annotatons    = val_annotatons.sample(30).reset_index(drop=True)
    # Build datasets
    # Without transforms for now
    train_dataset = EchoDataset(train_annotations, dataset_root, get_train_transforms())
    val_dataset = EchoDataset(val_annotations, dataset_root, get_val_transforms())


    # For debugging processes -> to check that data is being loaded correctly into a dataset 
    # for i in range(len(train_dataset)):
    #     train_dataset[i]

    # SANITY CHECKS
    print("Train samples: ", len(train_dataset))
    print("Val samples: ", len(val_dataset))

    print("Train studies: ", train_dataset.annotations["uid_study"].nunique())
    print("Val studies: ", val_dataset.annotations["uid_study"].nunique())
    

    # Build dataloaders

    train_loader = build_dataloader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
            )
    
    val_loader = build_dataloader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # Build model

    model = build_faster_rcnn_model(num_classes=10) # 9 anatomical classes + background
    model.to(device)

    # Optimizer configuration
    params = [ p for p in model.parameters() if p.requires_grad ] # Only parameters that require grads

    optimizer = torch.optim.SGD(
        params,
        lr = 0.001,
        momentum = 0.9,
        weight_decay = 0.0005
    )

    # Add scheduler to reduce oscillations and improve convergence
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=5,   # every 5 epochs
    #     gamma=0.1      # reduce LR ×10
    # )

    # Trying new scheduler for overcoming plateu
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', # minimize val loss
        factor=0.5, # Reduce LR by half
        patience = 3,
        min_lr = 1e-6
    )

    
    
    # Save checkpointtensorboard_logs_coneremoved_1703_trans_scheduler_final
    checkpoint_dir = Path("experiments/prop_7k_3_5_2303")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):

        print(f"\n===== Epoch {epoch} =====\n")

        # Training step
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            writer
        )

        # Evaluation on val set
        val_metrics = evaluate_detector(
            model,
            val_loader,
            device
        )

        writer.add_scalars(
            "loss",
            {
                "train": train_metrics["loss"],
                "val": val_metrics["total_loss"]
            },
            epoch
        )

        writer.add_scalars(
            "loss_classifier",
            {
                "train": train_metrics["loss_classifier"],
                "val": val_metrics["loss_classifier"],
            },
            epoch,
        )   

        writer.add_scalars(
            "loss_box_reg",
            {
                "train": train_metrics["loss_box_reg"],
                "val": val_metrics["loss_box_reg"],
            },
            epoch,
        )

        writer.add_scalars(
            "loss_objectness",
            {
                "train": train_metrics["loss_objectness"],
                "val": val_metrics["loss_objectness"],
            },
            epoch,
        )

        writer.add_scalars(
            "loss_rpn_box_reg",
            {
                "train": train_metrics["loss_rpn_box_reg"],
                "val": val_metrics["loss_rpn_box_reg"],
            },
            epoch,
        )


        checkpoint_path = checkpoint_dir / f"prop_7k_3_5_2303_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        print("Checkpoint saved: ", checkpoint_path)

        # Step the scheduler
        #scheduler.step()
        scheduler.step(val_metrics["total_loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)


if __name__ == "__main__":
    main()

