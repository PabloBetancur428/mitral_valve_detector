import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD ALL EXPERIMENTS
# =========================

exp_root = Path("E:\Z2455862L\Desktop\Random_Programming\mitral_valve_det\mitral_valve_detector\experiments_gone")

rows = []

for exp_dir in exp_root.iterdir():

    summary_file = exp_dir / "summary.json"

    if not summary_file.exists():
        continue

    with open(summary_file, "r") as f:
        data = json.load(f)

    data["run_name"] = exp_dir.name
    rows.append(data)

df = pd.DataFrame(rows)

if len(df) == 0:
    print("No experiments found.")
    exit()

# =========================
# SCORE
# =========================

df["val_loss_inv"] = 1 / (df["best_val_loss"] + 1e-8)

df["score"] = (
    0.5 * df["best_match_precision"] +
    0 * df["best_label_precision"] +
    0.5 * df["val_loss_inv"]
)

df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

TOP_K = 8
top_df = df.head(TOP_K)

print(f"Top {TOP_K} models:")
print(top_df[["run_name", "best_match_precision", "best_label_precision", "best_val_loss", "score"]])

# =========================
# FIGURE SETUP (IMPORTANT FIX)
# =========================

fig, ax = plt.subplots(figsize=(12, 8))

# =========================
# PLOT ALL MODELS
# =========================

scatter = ax.scatter(
    df["best_match_precision"],
    df["best_label_precision"],
    c=df["score"]
)

# Highlight top models
ax.scatter(
    top_df["best_match_precision"],
    top_df["best_label_precision"],
    marker='x'
)

# =========================
# ANNOTATE TOP MODELS
# =========================

for i, row in top_df.iterrows():
    ax.text(
        row["best_match_precision"],
        row["best_label_precision"],
        f"{i}",
        fontsize=9
    )

# =========================
# AXES
# =========================

ax.set_xlabel("Match Precision")
ax.set_ylabel("Label Precision")
ax.set_title("Model Comparison: Match vs Label Precision")

# =========================
# LEGEND (FIXED POSITION)
# =========================

legend_text = "\n".join(
    [f"{i}: {row['run_name']}" for i, row in top_df.iterrows()]
)

# Place legend INSIDE figure space
fig.text(
    0.75, 0.5,   # <-- moved inside figure
    legend_text,
    fontsize=9,
    verticalalignment='center',
    bbox=dict(facecolor='white', alpha=0.8)
)

# Adjust layout to avoid clipping
plt.subplots_adjust(right=0.7)

# Optional colorbar
fig.colorbar(scatter, ax=ax, label="Score")

plt.show()