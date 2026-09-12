"""
MedBoard — tests/test_preprocessing.py
Unit tests for the preprocessing module (Phase 1).

Run with:
    python tests/test_preprocessing.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import torch
import numpy as np
from modules.preprocessing import MRIPreprocessor, PreprocessedImage, PreprocessedMask

# ─── Paths to real BRISC 2025 samples ────────────────────────────────────────
DATA_ROOT = Path("data/raw")
CLF_TRAIN  = DATA_ROOT / "classification_task" / "train"
SEG_TRAIN  = DATA_ROOT / "segmentation_task" / "train"

def get_sample_image(class_name: str = "glioma") -> Path:
    """Get the first image from a classification class folder."""
    folder = CLF_TRAIN / class_name
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    assert len(images) > 0, f"No images found in {folder}"
    return images[0]

def get_sample_pair() -> tuple[Path, Path]:
    """Get a matched image+mask pair from the segmentation task."""
    images = list((SEG_TRAIN / "images").glob("*.jpg")) + \
             list((SEG_TRAIN / "images").glob("*.png"))
    assert len(images) > 0, "No segmentation images found"
    img_path = images[0]
    # Mask has same name but .png extension
    mask_path = SEG_TRAIN / "masks" / (img_path.stem + ".png")
    if not mask_path.exists():
        # Some datasets use same extension for masks
        mask_path = SEG_TRAIN / "masks" / img_path.name
    assert mask_path.exists(), f"No matching mask for {img_path.name}"
    return img_path, mask_path


def test_classification_preprocessing():
    """Test image preprocessing in classification mode."""
    print("\n[TEST 1] Classification preprocessing")
    preprocessor = MRIPreprocessor()
    img_path = get_sample_image("glioma")

    result = preprocessor.preprocess_image(img_path, mode="classification")

    assert isinstance(result, PreprocessedImage), "Wrong return type"
    assert result.tensor.shape == (3, 224, 224), \
        f"Wrong shape: {result.tensor.shape}, expected (3, 224, 224)"
    assert result.tensor.dtype == torch.float32, \
        f"Wrong dtype: {result.tensor.dtype}"
    assert result.mode == "classification"
    assert len(result.original_size) == 2

    # Check ImageNet normalization was applied (values should go below 0 or above 1)
    has_negative = result.tensor.min().item() < 0
    assert has_negative, "ImageNet normalization not applied (no negative values found)"

    print(f"  ✅ Image: {img_path.name}")
    print(f"  ✅ Original size: {result.original_size}")
    print(f"  ✅ Tensor shape:  {tuple(result.tensor.shape)}")
    print(f"  ✅ Tensor range:  [{result.tensor.min():.3f}, {result.tensor.max():.3f}]")
    print(f"  ✅ Has negative values (ImageNet norm confirmed): {has_negative}")


def test_segmentation_preprocessing():
    """Test image preprocessing in segmentation mode."""
    print("\n[TEST 2] Segmentation preprocessing")
    preprocessor = MRIPreprocessor()
    img_path = get_sample_image("meningioma")

    result = preprocessor.preprocess_image(img_path, mode="segmentation")

    assert isinstance(result, PreprocessedImage)
    assert result.tensor.shape == (3, 256, 256), \
        f"Wrong shape: {result.tensor.shape}, expected (3, 256, 256)"
    assert result.mode == "segmentation"

    # [0,1] range — no ImageNet normalization
    assert result.tensor.min().item() >= 0.0, "Segmentation tensor has negative values"
    assert result.tensor.max().item() <= 1.0, "Segmentation tensor exceeds 1.0"

    print(f"  ✅ Image: {img_path.name}")
    print(f"  ✅ Tensor shape: {tuple(result.tensor.shape)}")
    print(f"  ✅ Tensor range: [{result.tensor.min():.3f}, {result.tensor.max():.3f}] (no ImageNet norm)")


def test_mask_preprocessing():
    """Test segmentation mask preprocessing."""
    print("\n[TEST 3] Mask preprocessing")
    preprocessor = MRIPreprocessor()
    img_path, mask_path = get_sample_pair()

    result = preprocessor.preprocess_mask(mask_path)

    assert isinstance(result, PreprocessedMask)
    assert result.tensor.shape == (1, 256, 256), \
        f"Wrong shape: {result.tensor.shape}, expected (1, 256, 256)"
    assert result.tensor.dtype == torch.float32

    unique_values = result.tensor.unique().tolist()
    assert all(v in [0.0, 1.0] for v in unique_values), \
        f"Mask has non-binary values: {unique_values}"

    tumor_pixels = result.tensor.sum().item()
    total_pixels = 256 * 256
    tumor_pct = 100 * tumor_pixels / total_pixels

    print(f"  ✅ Mask: {mask_path.name}")
    print(f"  ✅ Tensor shape: {tuple(result.tensor.shape)}")
    print(f"  ✅ Unique values: {unique_values} (binary confirmed)")
    print(f"  ✅ Tumor coverage: {tumor_pct:.1f}% of image")


def test_pair_preprocessing():
    """Test paired image+mask preprocessing."""
    print("\n[TEST 4] Paired image+mask preprocessing")
    preprocessor = MRIPreprocessor()
    img_path, mask_path = get_sample_pair()

    image, mask = preprocessor.preprocess_pair(img_path, mask_path)

    # Image and mask must have matching spatial dimensions
    assert image.tensor.shape[1:] == mask.tensor.shape[1:], \
        f"Spatial size mismatch: image {image.tensor.shape}, mask {mask.tensor.shape}"

    print(f"  ✅ Image tensor: {tuple(image.tensor.shape)}")
    print(f"  ✅ Mask tensor:  {tuple(mask.tensor.shape)}")
    print(f"  ✅ Spatial sizes match: {image.tensor.shape[1:]}")


def test_all_classes():
    """Verify preprocessing works on all 4 tumor classes."""
    print("\n[TEST 5] All 4 classes")
    preprocessor = MRIPreprocessor()
    classes = ["glioma", "meningioma", "pituitary", "no_tumor"]

    for cls in classes:
        img_path = get_sample_image(cls)
        result = preprocessor.preprocess_image(img_path, mode="classification")
        assert result.tensor.shape == (3, 224, 224)
        print(f"  ✅ {cls:15s} → shape {tuple(result.tensor.shape)}, "
              f"range [{result.tensor.min():.2f}, {result.tensor.max():.2f}]")


def test_tensor_to_numpy():
    """Test conversion back to displayable numpy array."""
    print("\n[TEST 6] Tensor → numpy display conversion")
    preprocessor = MRIPreprocessor()
    img_path = get_sample_image("pituitary")

    result = preprocessor.preprocess_image(img_path, mode="classification")
    arr = MRIPreprocessor.tensor_to_numpy_display(result.tensor)

    assert arr.ndim == 3 and arr.shape[2] == 3, f"Wrong shape: {arr.shape}"
    assert arr.dtype == np.uint8
    assert arr.min() >= 0 and arr.max() <= 255

    print(f"  ✅ Numpy array shape: {arr.shape}")
    print(f"  ✅ Dtype: {arr.dtype}")
    print(f"  ✅ Value range: [{arr.min()}, {arr.max()}]")


# ─── Run All Tests ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MedBoard — Phase 1: Preprocessing Tests")
    print("=" * 55)

    tests = [
        test_classification_preprocessing,
        test_segmentation_preprocessing,
        test_mask_preprocessing,
        test_pair_preprocessing,
        test_all_classes,
        test_tensor_to_numpy,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 55)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ✅ Phase 1 preprocessing — ALL TESTS PASSED")
    else:
        print("  ❌ Some tests failed — check errors above")
    print("=" * 55)
