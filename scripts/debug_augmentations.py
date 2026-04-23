import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pathlib import Path

from src.datasets.echo_dataset import EchoDataset
from src.datasets.transforms import get_train_transforms


def draw_boxes(image, boxes):
    """
    Draw bounding boxes on an image for visualization.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor [C,H,W]

    boxes : torch.Tensor
        Bounding boxes [N,4] in format [x1,y1,x2,y2]
    """

    # Convert tensor image to numpy for matplotlib
    img = image.permute(1, 2, 0).cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw each bounding box
    for box in boxes:

        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

    plt.show()

def main():

    # Path to annotation CSV
    annotations_path = Path(
        r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\AoVdetector\data\annotations_boxes\mitral_annotations_2502.csv"
    )

    # Path to dataset root
    dataset_root = Path(
        r"\\NAS3_Z\all\DB_TARTAGLIA\496_estudios_tartaglia_jmcrespi"
    )

    # Load annotations
    annotations = pd.read_csv(annotations_path)
    annotations = annotations.sample(20)
    annotations = annotations.reset_index(drop=True)

    # Create dataset WITH augmentations
    dataset = EchoDataset(
        annotations,
        dataset_root,
        transforms=get_train_transforms()
    )


    # Visualize several samples
    for i in range(5):

        image, target = dataset[i]

        boxes = target["boxes"]

        print("Sample:", i)
        print("Boxes:", boxes)

        draw_boxes(image, boxes)

if __name__ == "__main__":
    main()