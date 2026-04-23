from torch.utils.data import DataLoader
from pathlib import Path
from src.datasets.echo_dataset import EchoDataset

def collate_fn(batch):

    """
        Default collate Pytorch tries to stack tensors, 
        but object detection requires list of images and targets
            
        Parameters
        ----------
        batch : list
            List of tuples returned by the dataset:
            [(image1, target1), (image2, target2), ...]

        Returns
        -------
        images : list
            List of image tensors

        targets : list
            List of target dictionaries
    """

    images, targets = zip(*batch)

    return list(images), list(targets)


def build_dataloader(dataset, batch_size=2, shuffle=True, num_workers=0):
    """
        Pytorch dataloader for detectinon algorithms

        Parameters
        ----------
        dataset : Dataset
            EchoDataset instance

        batch_size : int
            Number of samples per batch

        shuffle: bool
        Whether to shuffle dataset

        num_workers : int
        Number of parallel loading workers

        Returns
        -------
        DataLoader
    """

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader

if __name__ == "__main__":


    annotations = Path(r"E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\AoVdetector\data\annotations_boxes\mitral_annotations_2502.csv")
    root = Path(r"\\NAS3_Z\all\DB_TARTAGLIA\496_estudios_tartaglia_jmcrespi")

    dataset = EchoDataset(annotations, root)

    loader = build_dataloader(dataset, batch_size=2)

    images, targets = next(iter(loader))

    print("Batch size:", len(images))
    print("Image shape:", images[0].shape)
    print("Boxes:", targets[0]["boxes"])
    print("Boxes:", targets[1]["boxes"])