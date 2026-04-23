import torch
from tqdm import tqdm
import time


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    writer=None,
    print_freq=50,
    debug_timing=False
):
    """
    Train the model for ONE epoch.

    This function:
    - Performs forward + backward passes
    - Accumulates losses WITHOUT unnecessary GPU-CPU sync
    - Returns averaged losses for logging (TensorBoard compatible)

    Parameters
    ----------
    model : torch.nn.Module
        Faster R-CNN model

    dataloader : DataLoader
        Training dataloader

    optimizer : torch.optim
        Optimizer (Adam, SGD, etc.)

    device : torch.device
        CPU or CUDA

    epoch : int
        Current epoch number

    writer : SummaryWriter (optional)
        TensorBoard writer (NOT used here directly, only returns values)

    print_freq : int
        Frequency of progress updates

    debug_timing : bool
        If True → prints timing per batch (for profiling)
    """

    # Set model to training mode (enables gradients, dropout, etc.)
    model.train()

    # --- Running accumulators (kept as tensors to avoid sync) ---
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn = 0.0

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (images, targets) in enumerate(pbar):

        # =========================
        # 1. DATA TRANSFER
        # =========================
        if debug_timing:
            t0 = time.time()

        # Move images to GPU
        images = [img.to(device, non_blocking=True) for img in images]

        # Move targets (dict of tensors) to GPU
        targets = [
            {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        if debug_timing:
            t1 = time.time()

        # =========================
        # 2. FORWARD PASS
        # =========================
        loss_dict = model(images, targets)

        # Total loss = sum of all components
        losses = sum(loss_dict.values())

        if debug_timing:
            t2 = time.time()

        # =========================
        # 3. BACKWARD PASS
        # =========================
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if debug_timing:
            t3 = time.time()

        # =========================
        # 4. ACCUMULATE LOSSES (NO SYNC)
        # =========================
        running_loss += losses.detach()
        running_loss_classifier += loss_dict["loss_classifier"].detach()
        running_loss_box += loss_dict["loss_box_reg"].detach()
        # running_loss_objectness += loss_dict["loss_objectness"].detach()
        # running_loss_rpn += loss_dict["loss_rpn_box_reg"].detach()

        # =========================
        # 5. OPTIONAL DEBUG TIMING
        # =========================
        if debug_timing and batch_idx % print_freq == 0:
            print(
                f"LOAD+TRANSFER: {t1 - t0:.3f}s | "
                f"FORWARD: {t2 - t1:.3f}s | "
                f"BACKWARD: {t3 - t2:.3f}s"
            )

        # =========================
        # 6. LIGHT PROGRESS LOGGING
        # =========================
        if batch_idx % print_freq == 0:
            loss_value = losses.detach()
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}"  # single sync point
            })

    # =========================
    # 7. EPOCH SUMMARY
    # =========================

    num_batches = len(dataloader)

    # Convert to scalar ONLY once (important for performance)
    epoch_loss = (running_loss / num_batches).item()
    epoch_loss_classifier = (running_loss_classifier / num_batches).item()
    epoch_loss_box = (running_loss_box / num_batches).item()
    # epoch_loss_objectness = (running_loss_objectness / num_batches).item()
    # epoch_loss_rpn = (running_loss_rpn / num_batches).item()

    print(f"\nEpoch [{epoch}] completed | Avg Train Loss: {epoch_loss:.4f}\n")

    # Return dictionary → used in train.py for TensorBoard
    return {
        "loss": epoch_loss,
        "loss_classifier": epoch_loss_classifier,
        "loss_box_reg": epoch_loss_box,
        # "loss_objectness": epoch_loss_objectness,
        # "loss_rpn_box_reg": epoch_loss_rpn,
    }