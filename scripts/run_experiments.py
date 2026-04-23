import itertools
import subprocess

# =========================
# GRID DEFINITION
# =========================

# --- Optimizers ---
optimizers = ["sgd", "adamw"]

# --- Learning rates ---
sgd_lrs = [0.005, 0.01, 0.02]
adamw_lrs = [1e-4, 3e-4, 1e-3]

# --- Batch sizes ---
batch_sizes = [4, 8]

# --- Schedulers ---
schedulers = ["cosine", "step"]

# --- StepLR params ---
step_sizes = [5, 10]
gammas = [0.1, 0.5]

# =========================
# BUILD CONFIG LIST
# =========================

configs = []

for opt in optimizers:

    lrs = sgd_lrs if opt == "sgd" else adamw_lrs

    for lr, bs, sch in itertools.product(lrs, batch_sizes, schedulers):

        if sch == "cosine":
            configs.append({
                "optimizer": opt,
                "lr": lr,
                "batch_size": bs,
                "scheduler": sch,
                "step_size": None,
                "gamma": None
            })

        elif sch == "step":
            for step_size, gamma in itertools.product(step_sizes, gammas):
                configs.append({
                    "optimizer": opt,
                    "lr": lr,
                    "batch_size": bs,
                    "scheduler": sch,
                    "step_size": step_size,
                    "gamma": gamma
                })

# =========================
# RUN EXPERIMENTS
# =========================

total_runs = len(configs)
print(f"\nTotal experiments to run: {total_runs}\n")

for i, cfg in enumerate(configs):

    # =========================
    # SAFE UNIQUE RUN NAME
    # =========================
    if cfg["scheduler"] == "cosine":
        run_name = f"{cfg['optimizer']}_lr{cfg['lr']}_bs{cfg['batch_size']}_cosine"
    else:
        run_name = (
            f"{cfg['optimizer']}_lr{cfg['lr']}_bs{cfg['batch_size']}_"
            f"step_s{cfg['step_size']}_g{cfg['gamma']}"
        )

    print("=" * 60)
    print(f"[RUN {i+1}/{total_runs}] {run_name}")
    print("=" * 60)

    # =========================
    # BUILD COMMAND
    # =========================
    command = [
        "python", "-m", "scripts.train_while_gone",   # 🔥 IMPORTANT FIX
        "--lr", str(cfg["lr"]),
        "--batch_size", str(cfg["batch_size"]),
        "--optimizer", cfg["optimizer"],
        "--scheduler", cfg["scheduler"],
        "--run_name", run_name
    ]

    # Add StepLR params only if needed
    if cfg["scheduler"] == "step":
        command += [
            "--step_size", str(cfg["step_size"]),
            "--gamma", str(cfg["gamma"])
        ]

    # =========================
    # DEBUG PRINT (VERY IMPORTANT)
    # =========================
    print("COMMAND:", " ".join(command))

    # =========================
    # EXECUTE
    # =========================
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Run failed: {run_name}")
        continue

print("\nAll experiments completed.")