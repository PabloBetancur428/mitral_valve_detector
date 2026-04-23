from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_by_fname(df, dataset_root, fname_query):
    """
    Visualize all samples that match a given fname (partial or full).
    """

    # -----------------------------
    # FIND MATCHES
    # -----------------------------
    matches = df[df["fname"].astype(str).str.contains(fname_query)]

    if len(matches) == 0:
        print(f"[INFO] No matches found for: {fname_query}")
        return

    print(f"[INFO] Found {len(matches)} matching samples")

    # -----------------------------
    # LOOP THROUGH MATCHES
    # -----------------------------
    for idx, row in matches.iterrows():

        print("\n" + "="*60)
        print(f"Index: {idx}")
        print(f"Study: {row['uid_study']}")
        print(f"View: {row['view']}")
        print(f"Frame: {row['frame_index']}")
        print(f"Image: {row['processed_image']}")

        # -----------------------------
        # LOAD IMAGE
        # -----------------------------
        image_path = dataset_root / row["processed_image"]

        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}")
            continue

        image = np.load(image_path)

        # Ensure HWC
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)

        elif image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        # Normalize
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)

        # -----------------------------
        # PLOT
        # -----------------------------
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(image)

        # -----------------------------
        # DRAW BOXES
        # -----------------------------
        boxes = [
            (
                row["box1_x1"], row["box1_y1"],
                row["box1_x2"], row["box1_y2"],
                row["label1"], "red"
            ),
            (
                row["box2_x1"], row["box2_y1"],
                row["box2_x2"], row["box2_y2"],
                row["label2"], "blue"
            )
        ]

        for (x1, y1, x2, y2, label, color) in boxes:

            if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
                continue

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )

            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                f"{label}",
                color=color,
                fontsize=10,
                backgroundcolor="black"
            )

        ax.set_title(
            f"{row['view']} | frame {row['frame_index']}"
        )

        ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    annotations_path = Path(r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset\secure_annotations\annotations_fixed_1703.csv")
    dataset_root = Path(r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset\images")

    df = pd.read_csv(annotations_path)

    list_issues = ['1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006967.500.18.dcm', 
                   '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006968.797.20.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006970.282.25.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006970.313.26.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006971.579.28.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006972.657.30.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006965.907.11.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006965.907.11.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006965.907.11.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006981.313.55.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006986.844.72.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006988.125.78.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006989.282.82.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006990.407.85.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006990.657.86.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006991.704.87.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006991.782.88.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006992.16.89.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006993.438.94.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006994.782.99.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006997.0.5.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708006997.110.7.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708010892.704.46.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708010899.672.84.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708011385.219.84.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708015764.110.33.dcm', '1.2.826.0.1.3680043.2.135.738651.68951324.7.1708014141.0.80.dcm']

    for issue in list_issues:

        visualize_by_fname(df, dataset_root, issue)