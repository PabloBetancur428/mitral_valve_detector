import torch
from tqdm import tqdm


def evaluate_detector(model, dataloader, device):
    """
    Run validation and compute detection losses (efficient version)

    Notes
    -----
    - Uses model.train() because Faster R-CNN only returns losses in train mode
    - Avoids .item() inside loop (prevents GPU sync bottlenecks)
    - Converts to scalar ONLY once per epoch
    """

    # IMPORTANT: Faster R-CNN requires train mode to compute losses
    model.train()

    # --- Running accumulators (as tensors to avoid sync) ---
    total_loss = 0.0
    loss_classifier = 0.0
    loss_box_reg = 0.0
    loss_objectness = 0.0
    loss_rpn_box_reg = 0.0

    num_batches = 0

    with torch.no_grad():

        for images, targets in tqdm(dataloader, desc="Validation", leave=False):

            # =========================
            # 1. MOVE DATA TO DEVICE
            # =========================
            images = [img.to(device, non_blocking=True) for img in images]

            targets = [
                {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # =========================
            # 2. FORWARD PASS (LOSS MODE)
            # =========================
            loss_dict = model(images, targets)

            # Total loss (computed ONCE)
            losses = sum(loss_dict.values())

            # =========================
            # 3. ACCUMULATE (NO SYNC)
            # =========================
            total_loss += losses.detach()
            loss_classifier += loss_dict["loss_classifier"].detach()
            loss_box_reg += loss_dict["loss_box_reg"].detach()
            loss_objectness += loss_dict["loss_objectness"].detach()
            loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].detach()

            num_batches += 1

    # =========================
    # 4. FINAL AVERAGING (ONE SYNC)
    # =========================
    total_loss = (total_loss / num_batches).item()
    loss_classifier = (loss_classifier / num_batches).item()
    loss_box_reg = (loss_box_reg / num_batches).item()
    loss_objectness = (loss_objectness / num_batches).item()
    loss_rpn_box_reg = (loss_rpn_box_reg / num_batches).item()

    return {
        "total_loss": total_loss,
        "loss_classifier": loss_classifier,
        "loss_box_reg": loss_box_reg,
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }