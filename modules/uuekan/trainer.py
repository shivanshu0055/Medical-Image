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
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Use torch.amp modern API (PyTorch 2.0+) to avoid deprecation warnings
if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
    from torch.amp import GradScaler, autocast
    def get_autocast(device: torch.device):
        return autocast(device_type=device.type, enabled=(device.type == "cuda"))
    def get_scaler(device: torch.device):
        return GradScaler(device.type, enabled=(device.type == "cuda"))
else:
    from torch.cuda.amp import GradScaler, autocast
    def get_autocast(device: torch.device):
        return autocast(enabled=(device.type == "cuda"))
    def get_scaler(device: torch.device):
        return GradScaler(enabled=(device.type == "cuda"))

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
    """Monitors validation Dice score and saves best model checkpoints with full training state."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, checkpoint_path: str = "weights/uuekan_best.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_dice = -float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(
        self,
        val_dice: float,
        model: nn.Module,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        history: dict | None = None,
    ) -> bool:
        if val_dice > self.best_dice + self.min_delta:
            self.best_dice = val_dice
            self.best_epoch = epoch
            self.counter = 0

            # Save full training checkpoint dictionary
            checkpoint = {
                "epoch": epoch,
                "best_dice": val_dice,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "history": history or {},
            }
            torch.save(checkpoint, self.checkpoint_path)
            print(f"  [Checkpoint] Best val Dice improved to {val_dice:.4f} (Epoch {epoch:02d}) -> Saved to {self.checkpoint_path}")
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

        with get_autocast(device):
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

        # Print progress every 20 batches with flush
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                f"  Batch [{batch_idx+1:3d}/{len(train_loader)}] | "
                f"Loss: {loss.item():.4f} | Dice: {m['dice']:.4f} | IoU: {m['iou']:.4f}",
                flush=True
            )

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

            with get_autocast(device):
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
    resume: bool = True,
) -> Dict[str, List[float]]:
    """
    Main training routine for UUEKAN with full state checkpointing and resuming.
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
    scaler = get_scaler(device)
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    start_epoch = 1
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
        "lr": [],
    }

    # Auto-resume from checkpoint if requested and available
    if resume and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                if ckpt.get("optimizer_state_dict") is not None:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if ckpt.get("scheduler_state_dict") is not None:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

                saved_epoch = ckpt.get("epoch", 0)
                start_epoch = saved_epoch + 1
                early_stopping.best_dice = ckpt.get("best_dice", -float("inf"))
                early_stopping.best_epoch = saved_epoch
                if "history" in ckpt and isinstance(ckpt["history"], dict):
                    history = ckpt["history"]

                print(f"[Resume] Checkpoint loaded from {checkpoint_path}")
                print(f"  Previous Best Epoch : {saved_epoch:02d} | Best Val Dice: {early_stopping.best_dice:.4f}")
                print(f"  Resuming from Epoch  : {start_epoch:02d} to {epochs:02d} ({max(0, epochs - start_epoch + 1)} remaining)")
            else:
                model.load_state_dict(ckpt)
                print(f"[Resume] Loaded raw weights from {checkpoint_path}. Starting from Epoch 1.")
        except Exception as e:
            print(f"[Resume Warning] Could not resume from checkpoint ({e}). Starting fresh from Epoch 1.")

    if start_epoch > epochs:
        print(f"\n[Notice] Model has already completed all {epochs} epochs! Returning best model.")
        return history

    print(f"\nStarting UUEKAN Training (Epochs {start_epoch:02d} to {epochs:02d}, patience={patience}, grad_accum={grad_accum_steps})")
    print("=" * 75)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"\n>>> Epoch {epoch:02d}/{epochs:02d} (LR: {cur_lr:.2e})", flush=True)

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

        if early_stopping.step(val_dice, model, epoch, optimizer, scheduler, history):
            print(f"\n[Early Stopping] No improvement in validation Dice for {patience} epochs. Stopping.")
            break

    # Restore best weights
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        best_state = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
        print(f"\nRestoring best model weights from {checkpoint_path}...")
        model.load_state_dict(best_state)

    return history


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: CombinedUUEKANLoss | None = None,
    device: torch.device | None = None,
    threshold: float = 0.5,
    save_json_path: str | Path | None = None,
    base_dice: float = 0.8701,
    base_iou: float = 0.8021,
) -> Dict[str, Any]:
    """
    Evaluates UUEKAN across the entire test dataset.
    Accumulates per-sample and dataset-level metrics (Dice, IoU, Precision, Recall, Loss),
    prints a benchmark comparison against the base U-Net baseline, and optionally saves metrics to JSON.
    """
    if device is None:
        device = next(model.parameters()).device

    if criterion is None:
        criterion = CombinedUUEKANLoss()

    model.eval()
    total_loss = 0.0
    sub_losses: Dict[str, float] = {}

    all_dices: List[float] = []
    all_ious: List[float] = []
    all_precisions: List[float] = []
    all_recalls: List[float] = []
    total_samples = 0

    print("\n" + "=" * 75)
    print(f"Evaluating UUEKAN on Full Test Dataset ({len(test_loader.dataset)} samples)...")
    print("=" * 75)

    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size = images.size(0)
            total_samples += batch_size

            with get_autocast(device):
                outputs = model(images, auxiliary=True)
                loss, loss_dict = criterion(outputs, masks)

            main_pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            total_loss += loss.item() * batch_size

            for k, v in loss_dict.items():
                sub_losses[k] = sub_losses.get(k, 0.0) + v * batch_size

            pred_prob = torch.sigmoid(main_pred)
            pred_bin = (pred_prob > threshold).float()

            # Compute sample-wise metrics
            for i in range(batch_size):
                p = pred_bin[i].view(-1)
                t = masks[i].view(-1)

                intersection = (p * t).sum().item()
                union = (p + t).clamp(0, 1).sum().item()
                p_sum = p.sum().item()
                t_sum = t.sum().item()

                dice = (2.0 * intersection + 1e-6) / (p_sum + t_sum + 1e-6)
                iou = (intersection + 1e-6) / (union + 1e-6)
                prec = (intersection + 1e-6) / (p_sum + 1e-6)
                rec = (intersection + 1e-6) / (t_sum + 1e-6)

                all_dices.append(float(dice))
                all_ious.append(float(iou))
                all_precisions.append(float(prec))
                all_recalls.append(float(rec))

    mean_dice = float(np.mean(all_dices))
    std_dice = float(np.std(all_dices))
    mean_iou = float(np.mean(all_ious))
    std_iou = float(np.std(all_ious))
    mean_prec = float(np.mean(all_precisions))
    mean_rec = float(np.mean(all_recalls))
    mean_loss = float(total_loss / max(total_samples, 1))

    dice_gain = (mean_dice - base_dice) * 100.0
    iou_gain = (mean_iou - base_iou) * 100.0

    print(f"\n[Test Evaluation Results]")
    print(f"  Total Test Samples   : {total_samples}")
    print(f"  Test Loss            : {mean_loss:.4f}")
    print(f"  Test Dice (F1)       : {mean_dice:.4f} \u00b1 {std_dice:.4f} ({mean_dice*100:.2f}%)")
    print(f"  Test IoU             : {mean_iou:.4f} \u00b1 {std_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"  Test Precision       : {mean_prec:.4f} ({mean_prec*100:.2f}%)")
    print(f"  Test Recall          : {mean_rec:.4f} ({mean_rec*100:.2f}%)")
    print("\n[Benchmark Comparison with Base U-Net]")
    print(f"  Base U-Net Dice      : {base_dice*100:.2f}%")
    print(f"  UUEKAN Test Dice     : {mean_dice*100:.2f}% -> Gain: {dice_gain:+.2f}%")
    print(f"  Base U-Net IoU       : {base_iou*100:.2f}%")
    print(f"  UUEKAN Test IoU      : {mean_iou*100:.2f}% -> Gain: {iou_gain:+.2f}%")
    print("=" * 75)

    results = {
        "model_architecture": "UUEKAN",
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_samples": total_samples,
        "input_resolution": [512, 512, 3],
        "output_resolution": [512, 512, 1],
        "metrics": {
            "test_loss_total": round(mean_loss, 4),
            "test_dice_score": round(mean_dice, 4),
            "test_dice_std": round(std_dice, 4),
            "test_dice_percentage": round(mean_dice * 100.0, 2),
            "test_iou_score": round(mean_iou, 4),
            "test_iou_std": round(std_iou, 4),
            "test_iou_percentage": round(mean_iou * 100.0, 2),
            "test_precision": round(mean_prec, 4),
            "test_recall": round(mean_rec, 4),
            "base_unet_dice": round(base_dice, 4),
            "gain_over_unet_dice_pct": round(dice_gain, 2),
            "base_unet_iou": round(base_iou, 4),
            "gain_over_unet_iou_pct": round(iou_gain, 2),
            "benchmark_rating": "Superior" if mean_dice > base_dice else "Comparable",
        },
    }

    if save_json_path is not None:
        save_path = Path(save_json_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Test metrics successfully saved to: {save_path}")

    return results

