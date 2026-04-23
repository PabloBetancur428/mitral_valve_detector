from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    # ---- PATH TO YOUR GENERATED ANNOTATION CSV ----
    annotations_csv_path = Path(
        r"e:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\data\processed_cone_dataset_prop\annotations_propagated_fixed.csv"
    )

    if not annotations_csv_path.exists():
        print("[ERROR] Annotation CSV not found.")
        sys.exit(1)

    print("[INFO] Loading annotation CSV...")
    df = pd.read_csv(annotations_csv_path)

    if df.empty:
        print("[INFO] CSV is empty.")
        sys.exit(0)

    # ---------------------------------------------------------
    # Remove discarded rows (frame_index == -1)
    # ---------------------------------------------------------
    if "frame_index" in df.columns:
        df = df[df["frame_index"] >= 0].reset_index(drop=True)

    print(f"[INFO] Total valid annotated frames: {len(df)}")

    if "view" not in df.columns:
        print("[ERROR] 'view' column not found in CSV.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 1️⃣ Frames per view
    # ---------------------------------------------------------
    frames_per_view = df["view"].value_counts().sort_index()

    # ---------------------------------------------------------
    # 2️⃣ Unique studies per view (based on uid_study)
    # ---------------------------------------------------------
    studies_per_view = (
        df.groupby("view")["uid_study"]
        .nunique()
        .sort_index()
    )

    # ---------------------------------------------------------
    # 3️⃣ Unique study+file combinations per view
    # ---------------------------------------------------------
    unique_files_per_view = (
        df[["view", "uid_study", "fname"]]
        .drop_duplicates()
        .groupby("view")
        .size()
        .sort_index()
    )

    # ---------------------------------------------------------
    # PRINT SUMMARY
    # ---------------------------------------------------------
    print("\n================ SUMMARY PER VIEW ================")

    for view in sorted(frames_per_view.index):

        n_frames = frames_per_view.get(view, 0)
        n_studies = studies_per_view.get(view, 0)
        n_files = unique_files_per_view.get(view, 0)

        print(f"\nView: {view}")
        print(f"  Annotated frames      : {n_frames}")
        print(f"  Unique studies        : {n_studies}")
        print(f"  Unique study+files    : {n_files}")

    print("\n==================================================")

    # ---------------------------------------------------------
    # OPTIONAL: Plot Frames vs Studies
    # ---------------------------------------------------------
    summary_df = pd.DataFrame({
        "Frames": frames_per_view,
        "Unique Studies": studies_per_view
    })

    summary_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Annotated Frames vs Unique Studies per View")
    plt.xlabel("View")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()