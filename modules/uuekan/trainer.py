"""
MedBoard — modules/uuekan/trainer.py
Dedicated Training and Evaluation Pipeline for UUEKAN.

Features:
  - Automatic Mixed Precision (AMP FP16) for fast training and low VRAM
  - Gradient accumulation to achieve effective batch size = 12 on Colab T4
  - Comprehensive metric tracking (Dice, IoU, Precision, Recall)
  - Multi-component loss logging (Seg, Boundary, Uncertainty)
  - Early stopping and model checkpointing
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Tuple, List, Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .losses import CombinedUUEKANLoss


def compute_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Computes Dice, IoU, Precision, and Recall."""
    with torch.no_grad():
        pred_prob = torch.sigmoid(pred_logits)
        pred_bin = (pred_prob > threshold).float()

        p = pred_bin.view(-1)
        t = target.view(-1)

        intersection = (p * t).sum().item()
        union = (p + t).clamp(0, 1).sum().item()
        p_sum = p.sum().item()
        t_sum = t.sum().item()

        dice = (2.0 * intersection + eps) / (p_sum + t_sum + eps)
        iou = (intersection + eps) / (union + eps)
        precision = (intersection + eps) / (p_sum + eps)
        recall = (intersection + eps) / (t_sum + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


class EarlyStopping:
    """Monitors validation Dice score and saves best model weights."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, checkpoint_path: str = "weights/uuekan_best.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_dice = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_dice: float, model: nn.Module) -> bool:
        if val_dice > self.best_dice + self.min_delta:
            self.best_dice = val_dice
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  [Checkpoint] Best val Dice improved to {val_dice:.4f} -> Saved to {self.checkpoint_path}")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedUUEKANLoss,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int = 3,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Runs one training epoch with gradient accumulation and AMP."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    sub_losses: Dict[str, float] = {}

    optimizer.zero_grad()

    for batch_idx, (images, masks, _) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=torch.cuda.is_available()):
            # Forward pass with auxiliary heads enabled
            outputs = model(images, auxiliary=True)
            loss, loss_dict = criterion(outputs, masks)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Track metrics using main output
        main_pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        m = compute_metrics(main_pred, masks)

        total_loss += loss.item()
        total_dice += m["dice"]
        total_iou += m["iou"]

        for k, v in loss_dict.items():
            sub_losses[k] = sub_losses.get(k, 0.0) + v

    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    avg_subs = {k: v / num_batches for k, v in sub_losses.items()}

    return avg_loss, avg_dice, avg_iou, avg_subs


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: CombinedUUEKANLoss,
    device: torch.device,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Runs validation."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    sub_losses: Dict[str, float] = {}

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images, auxiliary=True)
                loss, loss_dict = criterion(outputs, masks)

            main_pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            m = compute_metrics(main_pred, masks)

            total_loss += loss.item()
            total_dice += m["dice"]
            total_iou += m["iou"]

            for k, v in loss_dict.items():
                sub_losses[k] = sub_losses.get(k, 0.0) + v

    num_batches = len(val_loader)
    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    avg_iou = total_iou / max(num_batches, 1)
    avg_subs = {k: v / max(num_batches, 1) for k, v in sub_losses.items()}

    return avg_loss, avg_dice, avg_iou, avg_subs


def train_uuekan(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    min_lr: float = 1e-5,
    grad_accum_steps: int = 3,
    patience: int = 10,
    checkpoint_path: str = "weights/uuekan_best.pth",
    device_str: str = "cuda",
) -> Dict[str, List[float]]:
    """
    Main training routine for UUEKAN.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Training UUEKAN on: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    criterion = CombinedUUEKANLoss()
    scaler = GradScaler(enabled=torch.cuda.is_available())
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
        "lr": [],
    }

    print(f"\nStarting UUEKAN Training ({epochs} epochs, patience={patience}, grad_accum={grad_accum_steps})")
    print("=" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_dice, train_iou, train_subs = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, grad_accum_steps=grad_accum_steps
        )

        val_loss, val_dice, val_iou, val_subs = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["lr"].append(cur_lr)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.0f}s] (LR: {cur_lr:.2e}) | "
            f"Train Loss: {train_loss:.4f} (Dice: {train_dice:.4f}, IoU: {train_iou:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Dice: {val_dice:.4f}, IoU: {val_iou:.4f})"
        )

        if early_stopping.step(val_dice, model):
            print(f"\n[Early Stopping] No improvement in validation Dice for {patience} epochs. Stopping.")
            break

    # Restore best weights
    if os.path.exists(checkpoint_path):
        print(f"\nRestoring best model weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return history
