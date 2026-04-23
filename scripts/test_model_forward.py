import torch
from pathlib import Path
from src.models.faster_rcnn_model import build_faster_rcnn_model
from src.training.dataloader import build_dataloader
from src.datasets.echo_dataset import EchoDataset


def main():

    annotations = Path(r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\AoVdetector\data\annotations_boxes\mitral_annotations_2502.csv")
    root = Path(r"\\NAS3_Z\all\DB_TARTAGLIA\496_estudios_tartaglia_jmcrespi")

    dataset = EchoDataset(annotations, root)
    dataloader = build_dataloader(dataset, batch_size=2)

    model = build_faster_rcnn_model(num_classes=3)

    model.train()

    images, targets = next(iter(dataloader))

    losses = model(images, targets)

    print("Losses")
    print(losses)



if __name__ == "__main__":
    main()