"""
MedBoard — modules/preprocessing.py
Phase 1: Image Preprocessing Module

Handles loading, resizing, and normalizing MRI images and segmentation masks
for downstream classification and segmentation models.

Design decisions:
  - Two modes: 'classification' (224x224, ImageNet stats) and
    'segmentation' (256x256, [0,1] normalization).
  - ImageNet stats used for classification because EfficientNet-B0 is
    pretrained on ImageNet — matching the training distribution matters.
  - [0,1] normalization for segmentation because U-Net is trained from
    scratch; ImageNet stats would be meaningless.
  - Masks use NEAREST resize to preserve binary values (0 or 1 only).
  - All sizes and stats are read from configs/config.yaml — nothing hardcoded.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ─── Config Loader ───────────────────────────────────────────────────────────

def _load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the central YAML config. Resolves relative to project root."""
    config_path = Path(config_path)
    if not config_path.exists():
        # Try relative to this file's location (two levels up)
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─── Output Dataclasses ──────────────────────────────────────────────────────

@dataclass
class PreprocessedImage:
    """Output of preprocessing an MRI image slice."""
    tensor: torch.Tensor          # Shape: (3, H, W), dtype: float32
    original_size: tuple[int, int]  # (original_H, original_W)
    mode: str                     # 'classification' or 'segmentation'
    image_path: str               # Source file path (for traceability)

    def __post_init__(self):
        assert self.tensor.ndim == 3 and self.tensor.shape[0] == 3, \
            f"Expected tensor shape (3, H, W), got {self.tensor.shape}"
        assert self.mode in ("classification", "segmentation"), \
            f"Mode must be 'classification' or 'segmentation', got '{self.mode}'"


@dataclass
class PreprocessedMask:
    """Output of preprocessing a binary segmentation mask."""
    tensor: torch.Tensor          # Shape: (1, H, W), dtype: float32, values: {0.0, 1.0}
    original_size: tuple[int, int]  # (original_H, original_W)
    mask_path: str                # Source file path (for traceability)

    def __post_init__(self):
        assert self.tensor.ndim == 3 and self.tensor.shape[0] == 1, \
            f"Expected tensor shape (1, H, W), got {self.tensor.shape}"


# ─── Preprocessor Class ──────────────────────────────────────────────────────

class MRIPreprocessor:
    """
    Preprocesses brain MRI images and segmentation masks for MedBoard.

    Usage:
        preprocessor = MRIPreprocessor()

        # For classification model input:
        result = preprocessor.preprocess_image(image_path, mode='classification')

        # For segmentation model input:
        result = preprocessor.preprocess_image(image_path, mode='segmentation')

        # For segmentation mask:
        mask = preprocessor.preprocess_mask(mask_path)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = _load_config(config_path)
        pp = cfg["preprocessing"]

        self.size_clf = pp["image_size_clf"]       # 224 for EfficientNet-B0
        self.size_seg = pp["image_size_seg"]       # 256 for U-Net

        # ImageNet stats — used ONLY for classification (pretrained backbone)
        self.imagenet_mean = pp["normalize_mean"]  # [0.485, 0.456, 0.406]
        self.imagenet_std  = pp["normalize_std"]   # [0.229, 0.224, 0.225]

        # Build transform pipelines
        self._clf_transform = self._build_clf_transform()
        self._seg_transform = self._build_seg_transform()
        self._mask_transform = self._build_mask_transform()

    # ── Transform Builders ───────────────────────────────────────────────────

    def _build_clf_transform(self) -> transforms.Compose:
        """
        Classification transform pipeline:
          Resize → CenterCrop → ToTensor → ImageNet Normalize
        Matches EfficientNet-B0 training preprocessing from timm.
        """
        return transforms.Compose([
            transforms.Resize((self.size_clf, self.size_clf),
                               interpolation=transforms.InterpolationMode.BILINEAR,
                               antialias=True),
            transforms.ToTensor(),                 # Converts [0,255] PIL → [0,1] tensor
            transforms.Normalize(mean=self.imagenet_mean,
                                  std=self.imagenet_std),
        ])

    def _build_seg_transform(self) -> transforms.Compose:
        """
        Segmentation transform pipeline:
          Resize → ToTensor → [0,1] range (no ImageNet normalization)
        U-Net is trained from scratch so ImageNet stats are not used.
        """
        return transforms.Compose([
            transforms.Resize((self.size_seg, self.size_seg),
                               interpolation=transforms.InterpolationMode.BILINEAR,
                               antialias=True),
            transforms.ToTensor(),                 # Converts [0,255] PIL → [0,1] tensor
            # No Normalize — [0,1] is fine for U-Net trained from scratch
        ])

    def _build_mask_transform(self) -> transforms.Compose:
        """
        Mask transform pipeline:
          Resize (NEAREST) → ToTensor → Threshold to binary
        NEAREST interpolation is critical — bilinear would create intermediate
        values (e.g. 0.5) in a binary mask, corrupting the ground truth.
        """
        return transforms.Compose([
            transforms.Resize((self.size_seg, self.size_seg),
                               interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),                 # Converts PIL → [0,1] tensor
        ])

    # ── Public API ───────────────────────────────────────────────────────────

    def preprocess_image(
        self,
        image_path: str | Path,
        mode: str = "classification",
    ) -> PreprocessedImage:
        """
        Load and preprocess a single MRI image slice.

        Args:
            image_path: Path to the .jpg or .png MRI image.
            mode: 'classification' (224x224, ImageNet norm) or
                  'segmentation' (256x256, [0,1] norm).

        Returns:
            PreprocessedImage with tensor shape (3, H, W).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and ensure 3-channel RGB
        # BRISC 2025 images are JPEG but some may be grayscale ('L' mode)
        img = Image.open(image_path).convert("RGB")
        original_size = (img.height, img.width)

        if mode == "classification":
            tensor = self._clf_transform(img)
        elif mode == "segmentation":
            tensor = self._seg_transform(img)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'classification' or 'segmentation'.")

        return PreprocessedImage(
            tensor=tensor,
            original_size=original_size,
            mode=mode,
            image_path=str(image_path),
        )

    def preprocess_mask(self, mask_path: str | Path) -> PreprocessedMask:
        """
        Load and preprocess a binary segmentation mask.

        Args:
            mask_path: Path to the .png binary mask.

        Returns:
            PreprocessedMask with tensor shape (1, H, W), values in {0.0, 1.0}.
        """
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load as grayscale — masks are single-channel binary images
        mask = Image.open(mask_path).convert("L")
        original_size = (mask.height, mask.width)

        tensor = self._mask_transform(mask)

        # Binarize: threshold at 0.5 to ensure values are exactly {0.0, 1.0}
        # (NEAREST resize should preserve binary values, but we enforce it)
        tensor = (tensor > 0.5).float()

        return PreprocessedMask(
            tensor=tensor,
            original_size=original_size,
            mask_path=str(mask_path),
        )

    def preprocess_pair(
        self,
        image_path: str | Path,
        mask_path: str | Path,
    ) -> tuple[PreprocessedImage, PreprocessedMask]:
        """
        Convenience method: preprocess an image + mask pair for segmentation training.

        Returns:
            (PreprocessedImage in segmentation mode, PreprocessedMask)
        """
        image = self.preprocess_image(image_path, mode="segmentation")
        mask = self.preprocess_mask(mask_path)
        return image, mask

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def tensor_to_numpy_display(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a preprocessed image tensor back to a displayable uint8 numpy array.
        Useful for visualization in Streamlit (st.image expects HWC uint8).

        Args:
            tensor: (3, H, W) or (1, H, W) float32 tensor.

        Returns:
            (H, W, C) or (H, W) uint8 numpy array in [0, 255].
        """
        t = tensor.clone().detach().cpu()

        if t.shape[0] == 3:
            # Reverse ImageNet normalization for display
            # (approximate — exact reversal only if normalized with ImageNet stats)
            t = t.permute(1, 2, 0).numpy()          # (H, W, 3)
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)  # Re-normalize to [0,1]
            t = (t * 255).astype(np.uint8)
        elif t.shape[0] == 1:
            t = t.squeeze(0).numpy()                 # (H, W)
            t = (t * 255).astype(np.uint8)

        return t


# ─── Module-level Convenience Functions ──────────────────────────────────────

def preprocess_for_classification(image_path: str | Path) -> torch.Tensor:
    """Quick one-liner: preprocess an image for the classification model."""
    p = MRIPreprocessor()
    result = p.preprocess_image(image_path, mode="classification")
    return result.tensor


def preprocess_for_segmentation(image_path: str | Path) -> torch.Tensor:
    """Quick one-liner: preprocess an image for the segmentation model."""
    p = MRIPreprocessor()
    result = p.preprocess_image(image_path, mode="segmentation")
    return result.tensor
