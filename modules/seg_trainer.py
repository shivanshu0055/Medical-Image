"""
MedBoard — modules/seg_trainer.py
Phase 2: U-Net Training Loop

This file contains the full training loop for the segmentation model.

Key concepts used here:
  - Dice + BCE loss: handles the class imbalance problem (tumor is tiny)
  - Mixed precision (FP16): uses half-precision floats to save VRAM
  - Early stopping: stops training if val loss stops improving (avoids overfitting)
  - Checkpointing: saves the best model weights to disk during training
  - Dice score: our evaluation metric (0=no overlap, 1=perfect match)
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
#  Loss Function: Dice + Binary Cross-Entropy (BCE)
#
#  WHY DICE LOSS?
#    The tumor occupies only ~6% of the image. If we used plain BCE,
#    a model that predicts "no tumor everywhere" would still get 94% accuracy!
#    That's useless for us. Dice loss directly measures overlap between the
#    predicted mask and the true mask — it penalizes missing the tumor heavily.
#
#  WHY COMBINE WITH BCE?
#    Dice loss alone can be unstable early in training (division by near-zero).
#    BCE provides a stable gradient from the start. Combined, they complement
#    each other: BCE for stable early training, Dice for precision later.
#
#  Combined loss = 0.5 * BCE + 0.5 * Dice
# ─────────────────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross-Entropy Loss.

    Dice loss: 1 - (2 * intersection) / (prediction_sum + target_sum + eps)
    BCE loss:  standard pixel-wise binary cross-entropy
    Combined:  0.5 * BCE + 0.5 * Dice
    """

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Small value added to numerator and denominator to avoid
                    division by zero. Also called epsilon.
        """
        super().__init__()
        self.smooth = smooth
        # BCEWithLogitsLoss combines sigmoid + BCE in one step.
        # This is numerically more stable than applying sigmoid separately.
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Raw model output (B, 1, H, W) — before sigmoid.
            targets: Ground truth binary masks (B, 1, H, W) — values {0.0, 1.0}.

        Returns:
            Combined scalar loss value.
        """
        # BCE loss (works on raw logits)
        bce_loss = self.bce(logits, targets)

        # Dice loss (needs probabilities, so apply sigmoid first)
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions for easier calculation
        # (B, 1, H, W) → (B, H*W)
        probs_flat   = probs.view(probs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        # Dice score formula:
        # intersection = number of pixels correctly predicted as tumor
        # union = total tumor pixels in prediction + total in target
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / \
                     (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)

        dice_loss = 1.0 - dice_score.mean()   # loss = 1 - score (we minimize loss)

        return 0.5 * bce_loss + 0.5 * dice_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation Metric: Dice Score
#
#  Dice Score = 2 * |Predicted ∩ Ground Truth| / (|Predicted| + |Ground Truth|)
#
#  Range: 0.0 (no overlap) to 1.0 (perfect overlap)
#  Target: > 0.75 is good, > 0.85 is excellent for brain tumor segmentation
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Compute Dice score between predicted and ground truth masks.

    Args:
        pred_mask: Model output (B, 1, H, W) — raw logits OR probabilities.
        true_mask: Ground truth (B, 1, H, W) — binary {0, 1}.
        threshold: Threshold to binarize predictions.
        smooth:    Epsilon to avoid division by zero.

    Returns:
        Mean Dice score across the batch (float, range [0, 1]).
    """
    # Convert logits/probs to binary mask
    if pred_mask.min() < 0 or pred_mask.max() > 1:
        pred_mask = torch.sigmoid(pred_mask)

    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask  > threshold).float()

    # Flatten
    pred_flat = pred_binary.view(pred_binary.shape[0], -1)
    true_flat = true_binary.view(true_binary.shape[0], -1)

    intersection = (pred_flat * true_flat).sum(dim=1)
    score = (2.0 * intersection + smooth) / \
            (pred_flat.sum(dim=1) + true_flat.sum(dim=1) + smooth)

    return score.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
#  Early Stopping
#
#  We stop training if the validation loss hasn't improved for `patience` epochs.
#  This prevents overfitting: the model memorizing training data instead of
#  learning general patterns.
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Also saves the best model weights so we can restore them at the end.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 checkpoint_path: str = "models/segmentation/unet_best.pth"):
        """
        Args:
            patience:         Stop after this many epochs with no improvement.
            min_delta:        Minimum change to count as improvement.
            checkpoint_path:  Where to save the best model weights.
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_loss    = float("inf")
        self.counter      = 0          # how many epochs without improvement
        self.should_stop  = False      # flip to True when we decide to stop

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Call after each epoch.

        Args:
            val_loss: Validation loss for this epoch.
            model:    The model to save if this is the best epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement! Save the model and reset counter.
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return False   # don't stop
        else:
            # No improvement.
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True    # stop training
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  One Epoch: Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    scaler:      GradScaler,
    device:      torch.device,
) -> tuple[float, float]:
    """
    Run one full pass through the training data.

    Uses mixed precision (FP16) to save VRAM on RTX 2050.

    Returns:
        (average_loss, average_dice_score) for this epoch.
    """
    model.train()   # enable training mode (BatchNorm and Dropout behave differently)

    total_loss  = 0.0
    total_dice  = 0.0
    num_batches = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):
        # Move data to GPU
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()   # clear gradients from last step

        # ── Forward pass with FP16 (mixed precision) ──────────────────────
        # autocast automatically decides which ops to run in FP16 vs FP32
        # for best accuracy + speed. FP16 uses half the VRAM of FP32.
        with autocast():
            logits = model(images)         # (B, 1, 256, 256)
            loss   = criterion(logits, masks)

        # ── Backward pass (compute gradients) ─────────────────────────────
        # scaler handles FP16 gradient scaling to prevent underflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        with torch.no_grad():
            batch_dice = dice_score(logits.detach(), masks)

        total_loss += loss.item()
        total_dice += batch_dice

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch [{batch_idx+1:3d}/{num_batches}]  "
                  f"loss={loss.item():.4f}  dice={batch_dice:.4f}")

    return total_loss / num_batches, total_dice / num_batches


# ─────────────────────────────────────────────────────────────────────────────
#  One Epoch: Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation set (no gradient updates).

    Returns:
        (average_loss, average_dice_score) for the validation set.
    """
    model.eval()   # disable dropout/batchnorm training behaviour

    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():   # don't compute gradients (saves memory + time)
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with autocast():
                logits = model(images)
                loss   = criterion(logits, masks)

            total_loss += loss.item()
            total_dice += dice_score(logits, masks)

    n = len(loader)
    return total_loss / n, total_dice / n


# ─────────────────────────────────────────────────────────────────────────────
#  Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_segmentation(
    model:          nn.Module,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    epochs:         int   = 40,
    learning_rate:  float = 1e-4,
    patience:       int   = 7,
    checkpoint_path: str  = "models/segmentation/unet_best.pth",
    device_str:     str   = "cuda",
) -> dict:
    """
    Full training loop for the U-Net segmentation model.

    Args:
        model:           UNet model (from modules/unet.py).
        train_loader:    Training DataLoader.
        val_loader:      Validation DataLoader.
        epochs:          Maximum number of training epochs.
        learning_rate:   Adam optimizer learning rate.
        patience:        Early stopping patience.
        checkpoint_path: Where to save the best model.
        device_str:      'cuda' or 'cpu'.

    Returns:
        Dictionary with training history:
        {'train_loss': [...], 'val_loss': [...],
         'train_dice': [...], 'val_dice': [...]}
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model = model.to(device)

    # ── Optimizer: Adam ───────────────────────────────────────────────────────
    # Adam adapts the learning rate for each parameter individually.
    # weight_decay=1e-5 is L2 regularization — gently penalizes large weights.
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # ── Learning Rate Scheduler ───────────────────────────────────────────────
    # If val loss doesn't improve for 3 epochs, reduce LR by factor of 0.5.
    # This helps the model fine-tune in the later stages of training.
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    # ── Loss Function ─────────────────────────────────────────────────────────
    criterion = DiceBCELoss()

    # ── Mixed Precision Scaler ────────────────────────────────────────────────
    # Required for FP16 training — manages gradient scaling
    scaler = GradScaler()

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    # ── Training History ──────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    print(f"\nStarting training: {epochs} max epochs, patience={patience}")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Train ────────────────────────────────────────────────────────
        print(f"\nEpoch {epoch:02d}/{epochs}")
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # ── Validate ─────────────────────────────────────────────────────
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - epoch_start

        # ── Log results ───────────────────────────────────────────────────
        print(f"  Train → loss: {train_loss:.4f} | dice: {train_dice:.4f}")
        print(f"  Val   → loss: {val_loss:.4f}   | dice: {val_dice:.4f}   [{elapsed:.0f}s]")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        # ── Update scheduler ──────────────────────────────────────────────
        scheduler.step(val_loss)

        # ── Early stopping check ──────────────────────────────────────────
        stop = early_stopping.step(val_loss, model)
        if stop:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            print(f"Best val loss: {early_stopping.best_loss:.4f}")
            break

    # Restore best weights
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"\nBest weights restored from: {checkpoint_path}")

    print(f"\nTraining complete. Best val dice: {max(history['val_dice']):.4f}")
    return history
