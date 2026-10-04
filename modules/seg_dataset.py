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

        Augmentation pipeline (applied in order):
          1. Random horizontal flip  (p=0.5)
          2. Random vertical flip    (p=0.5)
          3. Random rotation         (±20 degrees)
          4. Random scale + crop     (0.85x–1.15x, p=0.3)
          5. Elastic deformation     (p=0.3) — the key addition
          6. Brightness jitter       (image only, ±20%)
          7. Contrast jitter         (image only, ±20%)
          8. Random gamma            (image only, 0.8–1.2, p=0.3)
          9. Gaussian noise          (image only, p=0.2)

        All spatial transforms are applied identically to image and mask.
        Intensity transforms are applied to the IMAGE ONLY.
        """
        # ── 1. Random horizontal flip (50% chance) ────────────────────────
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # ── 2. Random vertical flip (50% chance) ─────────────────────────
        # Brain MRI slices have no strict up/down convention in 2D extracts,
        # and this doubles our effective augmentation diversity.
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # ── 3. Random rotation (±20 degrees) ─────────────────────────────
        angle = random.uniform(-20, 20)
        image = TF.rotate(image, angle, fill=0)
        mask  = TF.rotate(mask,  angle, fill=0)

        # ── 4. Random scale + crop (p=0.3) ───────────────────────────────
        # Simulates varying zoom levels / slice positions.
        # We scale up/down slightly, then center-crop back to original size.
        if random.random() < 0.3:
            w, h = image.size
            scale_factor = random.uniform(0.85, 1.15)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = TF.resize(image, [new_h, new_w],
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True)
            mask  = TF.resize(mask,  [new_h, new_w],
                              interpolation=transforms.InterpolationMode.NEAREST)
            # Center-crop (or pad) back to original size
            image = TF.center_crop(image, [h, w])
            mask  = TF.center_crop(mask,  [h, w])

        # ── 5. Elastic deformation (p=0.3) ───────────────────────────────
        # This is the most impactful augmentation for medical segmentation.
        # It simulates natural tissue deformation / anatomical variability.
        # Implementation uses pure PyTorch — no external dependencies needed.
        if random.random() < 0.3:
            image, mask = self._elastic_deformation(image, mask,
                                                     alpha=80.0, sigma=10.0)

        # ── 6. Brightness jitter (image only, ±20%) ──────────────────────
        brightness_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_brightness(image, brightness_factor)

        # ── 7. Contrast jitter (image only, ±20%) ────────────────────────
        contrast_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_contrast(image, contrast_factor)

        # ── 8. Random gamma correction (image only, p=0.3) ───────────────
        # Simulates non-linear intensity variations across different MRI
        # scanners and acquisition protocols.
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            image = TF.adjust_gamma(image, gamma)

        # ── 9. Gaussian noise (image only, p=0.2) ────────────────────────
        # Adds robustness to noisy / low-quality scans.
        if random.random() < 0.2:
            image = self._add_gaussian_noise(image, std=0.02)

        return image, mask

    @staticmethod
    def _elastic_deformation(
        image: Image.Image,
        mask: Image.Image,
        alpha: float = 80.0,
        sigma: float = 10.0,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Apply elastic deformation to image and mask using the same
        random displacement field (Simard et al., 2003).

        Pure PyTorch implementation — no scipy/albumentations needed.

        Args:
            image: PIL Image (RGB).
            mask:  PIL Image (grayscale).
            alpha: Deformation intensity (higher = more distortion).
            sigma: Gaussian smoothing kernel sigma (higher = smoother warps).

        Returns:
            Deformed (image, mask) as PIL Images.
        """
        import numpy as np

        w, h = image.size

        # Generate random displacement fields
        rng = np.random.default_rng()
        dx = rng.standard_normal((h, w)).astype(np.float32)
        dy = rng.standard_normal((h, w)).astype(np.float32)

        # Smooth with a Gaussian kernel (implemented via 1D separable convolution)
        dx_t = torch.from_numpy(dx).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        dy_t = torch.from_numpy(dy).unsqueeze(0).unsqueeze(0)

        # Create 1D Gaussian kernel
        ksize = int(6 * sigma + 1) | 1  # ensure odd
        x_coord = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        gauss_1d = torch.exp(-0.5 * (x_coord / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Separable Gaussian blur: convolve along H, then along W
        kernel_h = gauss_1d.view(1, 1, -1, 1)  # (1, 1, K, 1)
        kernel_w = gauss_1d.view(1, 1, 1, -1)  # (1, 1, 1, K)
        pad_h = ksize // 2
        pad_w = ksize // 2

        dx_t = torch.nn.functional.pad(dx_t, [0, 0, pad_h, pad_h], mode='reflect')
        dx_t = torch.nn.functional.conv2d(dx_t, kernel_h)
        dx_t = torch.nn.functional.pad(dx_t, [pad_w, pad_w, 0, 0], mode='reflect')
        dx_t = torch.nn.functional.conv2d(dx_t, kernel_w)

        dy_t = torch.nn.functional.pad(dy_t, [0, 0, pad_h, pad_h], mode='reflect')
        dy_t = torch.nn.functional.conv2d(dy_t, kernel_h)
        dy_t = torch.nn.functional.pad(dy_t, [pad_w, pad_w, 0, 0], mode='reflect')
        dy_t = torch.nn.functional.conv2d(dy_t, kernel_w)

        # Scale by alpha
        dx_t = dx_t * alpha
        dy_t = dy_t * alpha

        # Build sampling grid: base grid + displacement
        # grid_sample expects grid in [-1, 1] range
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )
        # Normalize displacement to [-1, 1] scale
        grid_x = grid_x.unsqueeze(0).unsqueeze(0) + (dx_t / (w / 2))
        grid_y = grid_y.unsqueeze(0).unsqueeze(0) + (dy_t / (h / 2))

        # Combine into (1, H, W, 2) grid for grid_sample
        grid = torch.cat([grid_x, grid_y], dim=1)  # (1, 2, H, W)
        grid = grid.squeeze(0).permute(1, 2, 0)      # (H, W, 2)
        grid = grid.unsqueeze(0)                       # (1, H, W, 2)

        # Warp image (bilinear interpolation)
        img_tensor = TF.to_tensor(image).unsqueeze(0)  # (1, 3, H, W)
        img_warped = torch.nn.functional.grid_sample(
            img_tensor, grid, mode='bilinear', padding_mode='zeros',
            align_corners=True
        )
        image_out = TF.to_pil_image(img_warped.squeeze(0).clamp(0, 1))

        # Warp mask (nearest interpolation to preserve binary values)
        mask_tensor = TF.to_tensor(mask).unsqueeze(0)  # (1, 1, H, W)
        mask_warped = torch.nn.functional.grid_sample(
            mask_tensor, grid, mode='nearest', padding_mode='zeros',
            align_corners=True
        )
        mask_out = TF.to_pil_image(mask_warped.squeeze(0))

        return image_out, mask_out

    @staticmethod
    def _add_gaussian_noise(
        image: Image.Image,
        std: float = 0.02,
    ) -> Image.Image:
        """
        Add Gaussian noise to an image (simulates scanner noise).
        Applied in [0,1] tensor space, then converted back to PIL.

        Args:
            image: PIL Image (RGB).
            std:   Standard deviation of the noise.

        Returns:
            Noisy PIL Image.
        """
        tensor = TF.to_tensor(image)  # (3, H, W), [0, 1]
        noise = torch.randn_like(tensor) * std
        noisy = (tensor + noise).clamp(0, 1)
        return TF.to_pil_image(noisy)


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
