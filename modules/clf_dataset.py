"""
MedBoard — modules/clf_dataset.py
Phase 3: Classification Dataset & DataLoader Builder

Loads BRISC 2025 classification MRI images for 4 classes:
    0: glioma
    1: meningioma
    2: no_tumor
    3: pituitary

Features:
  - Stratified train/val split from the train folder (preserves class proportions)
  - Data augmentation for training (flips, slight rotation, color jitter)
  - Strict ImageNet normalization for EfficientNet-B0
  - Test loader creation for benchmark evaluation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEFAULT_CLASSES: list[str] = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]

# ImageNet statistics for EfficientNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_default_transforms(
    image_size: int = 224,
    is_train: bool = False,
    rotation_deg: int = 15,
) -> Callable:
    """
    Build torchvision transforms for classification.

    Args:
        image_size: Input image dimension (default: 224).
        is_train: If True, include augmentations (flip, rotation, color jitter).
        rotation_deg: Maximum rotation degrees for train augmentation.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation_deg),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class BRISCClassificationDataset(Dataset):
    """
    PyTorch Dataset for brain MRI tumor classification.

    Args:
        samples: List of tuples `(image_path, class_idx)`.
        transform: Callable torchvision transform.
    """

    def __init__(
        self,
        samples: Sequence[tuple[str | Path, int]],
        transform: Optional[Callable] = None,
    ):
        self.samples = [(str(path), int(label)) for path, label in samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]

        # Load MRI scan as RGB image (3 channels)
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            # Fallback basic tensor conversion if no transform provided
            image = transforms.ToTensor()(image)

        return image, label


def scan_dataset_dir(
    root_dir: str | Path,
    class_names: Sequence[str] = DEFAULT_CLASSES,
) -> list[tuple[str, int]]:
    """
    Scan a directory structured with class subfolders and return sample pairs.

    Expected structure:
        root_dir/
            glioma/
            meningioma/
            no_tumor/
            pituitary/

    Returns:
        List of (image_path, class_idx) tuples.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Classification directory not found: {root_path}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: list[tuple[str, int]] = []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_name, class_idx in class_to_idx.items():
        class_dir = root_path / class_name
        if not class_dir.is_dir():
            continue

        for entry in os.scandir(class_dir):
            if entry.is_file() and Path(entry.name).suffix.lower() in valid_exts:
                samples.append((entry.path, class_idx))

    return samples


def create_classification_dataloaders(
    train_dir: str | Path,
    test_dir: Optional[str | Path] = None,
    val_ratio: float = 0.15,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 2,
    seed: int = 42,
    class_names: Sequence[str] = DEFAULT_CLASSES,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and optional test DataLoaders.

    Args:
        train_dir: Directory containing training images organized by class subfolders.
        test_dir: Optional directory containing test images organized by class subfolders.
        val_ratio: Fraction of train images to reserve for validation (default: 0.15).
        batch_size: Batch size for loaders.
        image_size: Target square image dimension (default: 224).
        num_workers: DataLoader worker count.
        seed: Random seed for stratified splitting.
        class_names: Class subfolder names.

    Returns:
        tuple (train_loader, val_loader, test_loader)
    """
    train_samples = scan_dataset_dir(train_dir, class_names=class_names)
    if not train_samples:
        raise ValueError(f"No valid images found in train directory: {train_dir}")

    # Stratified Train/Val split to maintain balanced representation of all classes
    paths, labels = zip(*train_samples)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_data = list(zip(train_paths, train_labels))
    val_data = list(zip(val_paths, val_labels))

    train_transform = get_default_transforms(image_size=image_size, is_train=True)
    val_transform = get_default_transforms(image_size=image_size, is_train=False)

    train_dataset = BRISCClassificationDataset(train_data, transform=train_transform)
    val_dataset = BRISCClassificationDataset(val_data, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = None
    if test_dir is not None and Path(test_dir).exists():
        test_samples = scan_dataset_dir(test_dir, class_names=class_names)
        if test_samples:
            test_dataset = BRISCClassificationDataset(test_samples, transform=val_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

    return train_loader, val_loader, test_loader
