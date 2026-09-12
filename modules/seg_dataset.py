"""
MedBoard — modules/seg_dataset.py
Phase 2: Segmentation Dataset Loader

This file defines a PyTorch Dataset class for the segmentation task.

What is a Dataset class?
  PyTorch needs a standard interface to load data during training.
  You give it a Dataset object, it calls __getitem__(index) to get each
  sample and __len__() to know how many samples exist.
  The DataLoader wraps this to load data in batches, in parallel.

This dataset:
  - Reads image + mask file pairs from BRISC 2025 segmentation_task/
  - Calls MRIPreprocessor to resize/normalize each pair
  - Applies augmentation during training (flips, rotation, brightness)
  - Returns (image_tensor, mask_tensor) ready for the U-Net
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from modules.preprocessing import MRIPreprocessor


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for brain MRI segmentation using BRISC 2025.

    Loads paired (image, mask) samples from:
      segmentation_task/train/images/ + segmentation_task/train/masks/
    or
      segmentation_task/test/images/  + segmentation_task/test/masks/

    Returns:
        image_tensor: (3, 256, 256) float32 — normalized MRI image
        mask_tensor:  (1, 256, 256) float32 — binary tumor mask {0.0, 1.0}
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir:  str | Path,
        split: Literal["train", "val", "test"] = "train",
        augment: bool | None = None,
        config_path: str = "configs/config.yaml",
    ):
        """
        Args:
            images_dir:  Path to the folder containing MRI image files.
            masks_dir:   Path to the folder containing mask files.
            split:       'train', 'val', or 'test'.
                         Augmentation is ON for 'train', OFF for others.
            augment:     Override augmentation flag. If None, follows split.
            config_path: Path to config.yaml for preprocessing settings.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.split      = split
        self.preprocessor = MRIPreprocessor(config_path)

        # Augmentation is applied during training only
        # (we don't want to randomly flip test images — results must be consistent)
        if augment is None:
            self.augment = (split == "train")
        else:
            self.augment = augment

        # Collect all image file paths
        image_extensions = {".jpg", ".jpeg", ".png"}
        self.image_paths = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        # Match each image to its mask (same filename, .png extension)
        self.pairs = []  # list of (image_path, mask_path) tuples
        missing = []

        for img_path in self.image_paths:
            # Try .png mask first, then same extension
            mask_path = self.masks_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                mask_path = self.masks_dir / img_path.name
            if not mask_path.exists():
                missing.append(img_path.name)
                continue
            self.pairs.append((img_path, mask_path))

        if missing:
            print(f"[SegmentationDataset] Warning: {len(missing)} images have no matching mask.")

        print(f"[SegmentationDataset] {split}: {len(self.pairs)} image-mask pairs loaded "
              f"| augmentation={'ON' if self.augment else 'OFF'}")

    def __len__(self) -> int:
        """Total number of samples in this dataset."""
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return one (image, mask) pair.

        Args:
            index: Sample index (0 to len-1).

        Returns:
            image: (3, 256, 256) float32 tensor
            mask:  (1, 256, 256) float32 tensor with values {0.0, 1.0}
        """
        img_path, mask_path = self.pairs[index]

        # Load the image as a PIL Image (RGB)
        image = Image.open(img_path).convert("RGB")

        # Load the mask as grayscale (single channel)
        mask = Image.open(mask_path).convert("L")

        # ── Apply augmentation (training only) ───────────────────────────
        # IMPORTANT: image and mask must receive IDENTICAL transforms.
        # If we flip the image, we must flip the mask the same way —
        # otherwise the mask won't match the image anymore.
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # ── Preprocess image → tensor ─────────────────────────────────────
        # Use the segmentation pipeline: resize to 256×256, normalize to [0,1]
        preprocessed = self.preprocessor.preprocess_image(img_path, mode="segmentation")

        # Note: we re-open via preprocessor for consistency, but we need to
        # apply augmentation to PIL first, then convert to tensor manually
        # to keep image/mask augmentation in sync.
        image_tensor = self._pil_to_seg_tensor(image)

        # ── Preprocess mask → tensor ─────────────────────────────────────
        mask_tensor = self._pil_mask_to_tensor(mask)

        return image_tensor, mask_tensor

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _pil_to_seg_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convert a PIL image to a segmentation-mode tensor.
        Resize to 256×256, convert to float [0,1]. No ImageNet normalization.
        """
        seg_transform = transforms.Compose([
            transforms.Resize((256, 256),
                               interpolation=transforms.InterpolationMode.BILINEAR,
                               antialias=True),
            transforms.ToTensor(),   # [0,255] → [0.0,1.0] float32
        ])
        return seg_transform(image)

    def _pil_mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        """
        Convert a PIL mask to a binary tensor.
        Resize with NEAREST (preserves binary values), then threshold at 0.5.
        """
        mask_transform = transforms.Compose([
            transforms.Resize((256, 256),
                               interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),   # [0,255] → [0.0,1.0]
        ])
        tensor = mask_transform(mask)
        return (tensor > 0.5).float()   # enforce binary: {0.0, 1.0}

    def _apply_augmentation(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Apply random augmentation to BOTH image and mask simultaneously.

        All spatial transforms (flip, rotate) are applied identically to both.
        Color/brightness transforms are applied to the IMAGE ONLY —
        the mask should not change color, it's a binary label.
        """
        # ── Random horizontal flip (50% chance) ───────────────────────────
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)   # same flip on mask

        # ── Random rotation (±15 degrees) ─────────────────────────────────
        # fill=0 means the empty border after rotation is black (background)
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle, fill=0)
        mask  = TF.rotate(mask,  angle, fill=0)   # same rotation on mask

        # ── Random brightness and contrast (image only) ───────────────────
        # MRI brightness can vary between scanners, so training with
        # different brightness levels makes the model more robust.
        brightness_factor = random.uniform(0.8, 1.2)   # ±20%
        contrast_factor   = random.uniform(0.8, 1.2)   # ±20%
        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)
        # Note: mask is NOT modified by brightness/contrast — it's a binary label

        return image, mask


# ─── DataLoader Factory Functions ────────────────────────────────────────────

def get_seg_dataloaders(
    train_images_dir: str,
    train_masks_dir:  str,
    test_images_dir:  str,
    test_masks_dir:   str,
    batch_size:  int = 8,
    num_workers: int = 4,
    val_split:   float = 0.1,
    seed:        int = 42,
    config_path: str = "configs/config.yaml",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders for segmentation.

    Splits the training set into train (90%) and val (10%) by default.

    Args:
        train_images_dir: Path to segmentation_task/train/images/
        train_masks_dir:  Path to segmentation_task/train/masks/
        test_images_dir:  Path to segmentation_task/test/images/
        test_masks_dir:   Path to segmentation_task/test/masks/
        batch_size:       Images per batch (default 8 for 4GB VRAM).
        num_workers:      Parallel data loading workers.
        val_split:        Fraction of training data to use for validation.
        seed:             Random seed for reproducible splits.
        config_path:      Path to config.yaml.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import random_split

    # Full training dataset (augmentation ON)
    full_train = SegmentationDataset(
        train_images_dir, train_masks_dir,
        split="train", config_path=config_path,
    )

    # Split into train and validation subsets
    total = len(full_train)
    val_size   = int(total * val_split)
    train_size = total - val_size

    # Use a fixed seed so the split is the same every run
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)

    # Override augmentation for the val subset: validation should not be augmented.
    # We do this by setting the flag on the underlying dataset after splitting.
    # (random_split returns Subset objects wrapping the original dataset)
    # The simpler approach: create a separate val dataset with augment=False
    val_dataset = SegmentationDataset(
        train_images_dir, train_masks_dir,
        split="val", augment=False, config_path=config_path,
    )
    # Use only the val indices from our split
    val_dataset_subset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    # Test dataset (no augmentation)
    test_dataset = SegmentationDataset(
        test_images_dir, test_masks_dir,
        split="test", augment=False, config_path=config_path,
    )

    # pin_memory=True speeds up CPU→GPU transfer during training
    common_kwargs = dict(num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_subset,       batch_size=batch_size, shuffle=True,  **common_kwargs)
    val_loader   = DataLoader(val_dataset_subset, batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_dataset,       batch_size=batch_size, shuffle=False, **common_kwargs)

    print(f"DataLoaders ready:")
    print(f"  Train : {len(train_subset):4d} samples → {len(train_loader):3d} batches")
    print(f"  Val   : {len(val_dataset_subset):4d} samples → {len(val_loader):3d} batches")
    print(f"  Test  : {len(test_dataset):4d} samples → {len(test_loader):3d} batches")

    return train_loader, val_loader, test_loader
