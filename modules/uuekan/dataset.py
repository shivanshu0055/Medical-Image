"""
MedBoard — modules/uuekan/dataset.py
Dedicated Dataset Loader for UUEKAN at 512x512 resolution on BRISC.

Preserves exact binary mask values using nearest-neighbor interpolation,
and applies synchronized data augmentations during training.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple, List, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image


class UUEKANSegmentationDataset(Dataset):
    """
    PyTorch Dataset for BRISC MRI tumor segmentation at 512x512.
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        img_size: int = 512,
        split: Literal["train", "val", "test"] = "train",
        augment: bool | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.split = split
        self.augment = (split == "train") if augment is None else augment

        valid_exts = {".jpg", ".jpeg", ".png"}
        all_images = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in valid_exts])

        self.samples: List[Tuple[Path, Path]] = []
        for img_path in all_images:
            # Check matching mask with same stem
            for ext in [".png", ".jpg", ".jpeg"]:
                mask_candidate = self.masks_dir / f"{img_path.stem}{ext}"
                if mask_candidate.exists():
                    self.samples.append((img_path, mask_candidate))
                    break

        if len(self.samples) == 0:
            print(f"[Warning] No matching (image, mask) pairs found in {images_dir} and {masks_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentations(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation (-15 to 15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15.0, 15.0)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Subtle brightness / contrast adjustment (image only)
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            image = TF.adjust_brightness(image, factor)
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            image = TF.adjust_contrast(image, factor)

        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, mask_path = self.samples[idx]

        # Load RGB image and grayscale mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        if self.augment:
            image, mask = self._apply_augmentations(image, mask)

        # Convert to Tensor [0, 1]
        img_arr = np.array(image, dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.float32)

        # Binarize mask
        mask_arr = (mask_arr > 127.5).astype(np.float32)

        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # (3, H, W)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)    # (1, H, W)

        return img_tensor, mask_tensor, img_path.name


def get_uuekan_dataloaders(
    train_images_dir: str | Path,
    train_masks_dir: str | Path,
    test_images_dir: str | Path,
    test_masks_dir: str | Path,
    img_size: int = 512,
    batch_size: int = 4,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, val, and test dataloaders for UUEKAN.
    """
    full_train = UUEKANSegmentationDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        img_size=img_size,
        split="train",
        augment=True,
    )

    # Train / Val Split
    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=generator
    )

    # Validation dataset (without augmentation)
    val_dataset = UUEKANSegmentationDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        img_size=img_size,
        split="val",
        augment=False,
    )
    val_subset_clean = torch.utils.data.Subset(val_dataset, val_subset.indices)

    test_dataset = UUEKANSegmentationDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        img_size=img_size,
        split="test",
        augment=False,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset_clean,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[UUEKAN DataLoaders Ready]")
    print(f"  Resolution : {img_size}x{img_size}")
    print(f"  Train      : {len(train_subset)} samples ({len(train_loader)} batches of {batch_size})")
    print(f"  Val        : {len(val_subset_clean)} samples ({len(val_loader)} batches)")
    print(f"  Test       : {len(test_dataset)} samples ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader
