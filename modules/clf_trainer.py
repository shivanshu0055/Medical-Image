"""
MedBoard — modules/clf_trainer.py
Phase 3: Classification Model Training & Evaluation Loop

Handles training and evaluation of the EfficientNet-B0 classifier:
  - CrossEntropyLoss with optional label smoothing
  - FP16 mixed precision for fast training on RTX 2050 and Colab T4
  - Early stopping with patience based on validation loss
  - Cosine annealing learning rate schedule
  - Accuracy and Macro-F1 metric computation
  - Detailed evaluation function returning confusion matrix & classification report
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEFAULT_CLASSES: list[str] = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]


class EarlyStopping:
    """
    Early stops training when validation loss stops improving.
    Saves the best model checkpoint to disk.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        checkpoint_path: str | Path,
    ) -> bool:
        """
        Check if validation loss improved.

        Returns:
            True if early stopping condition reached, else False.
        """
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(
                    f"  Val loss improved: {self.best_loss:.4f} -> {val_loss:.4f}. Saving checkpoint..."
                )
            self.best_loss = val_loss
            self.counter = 0

            # Save checkpoint
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        tuple (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = scaler is not None and device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str] = DEFAULT_CLASSES,
) -> dict:
    """
    Evaluate the model on validation or test dataset.

    Returns:
        dict containing:
          - 'loss': float
          - 'accuracy': float
          - 'macro_f1': float
          - 'confusion_matrix': np.ndarray
          - 'classification_report': str
          - 'predictions': np.ndarray
          - 'targets': np.ndarray
    """
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    total = len(y_true)
    val_loss = running_loss / max(total, 1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "loss": val_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": y_pred,
        "targets": y_true,
    }


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    label_smoothing: float = 0.05,
    checkpoint_path: str | Path = "models/classification/efficientnet_b0_brisc.pth",
    device_str: str = "cuda",
    use_mixed_precision: bool = True,
    class_names: Sequence[str] = DEFAULT_CLASSES,
) -> dict:
    """
    Full training pipeline for the classification model.

    Args:
        model: EfficientNet-B0 classifier.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        epochs: Number of training epochs.
        learning_rate: Initial AdamW learning rate.
        weight_decay: L2 regularization penalty.
        patience: Epochs to wait for val loss improvement before stopping.
        label_smoothing: Label smoothing factor in CrossEntropyLoss.
        checkpoint_path: Destination path for best model weights.
        device_str: "cuda" or "cpu".
        use_mixed_precision: Whether to use FP16 on GPU.
        class_names: Names of classes.

    Returns:
        history dict with metrics per epoch.
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Training on device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # Cosine annealing schedules smoothly towards 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda") if (use_mixed_precision and device.type == "cuda") else None
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting training for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:02d}/{epochs:02d} [LR: {current_lr:.6f}]")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
        )
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%\n"
            f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}% | Val Macro-F1: {val_f1:.4f}"
        )

        if early_stopping(val_loss, model, checkpoint_path):
            print(f"\n[EarlyStopping] Triggered at epoch {epoch}. Stopping training.")
            print(f"Best val loss achieved: {early_stopping.best_loss:.4f}")
            break

    # Restore best checkpoint
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"\nBest weights restored from: {checkpoint_path}")

    best_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    best_f1 = max(history["val_f1"]) if history["val_f1"] else 0.0
    print(f"Training finished. Best Val Acc: {best_acc * 100:.2f}%, Best Val F1: {best_f1:.4f}")

    return history
