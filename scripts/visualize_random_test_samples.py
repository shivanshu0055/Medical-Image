"""
MedBoard — scripts/visualize_random_test_samples.py
Selects 10 random test images from the BRISC test split, runs U-Net segmentation,
computes slice-level Dice and IoU metrics, and plots side-by-side visual comparisons
against the ground truth masks.

Columns plotted per sample:
  1. Raw Brain MRI
  2. Ground Truth Mask
  3. U-Net Predicted Binary Mask
  4. Continuous Tumor Probability Map
  5. Color Overlay (Green=True Positive, Red=False Positive, Yellow=False Negative)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modules.unet import UNet


def compute_slice_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Computes Dice similarity and IoU (Jaccard) for a single binary slice."""
    p = pred_bin.flatten()
    t = gt_bin.flatten()
    intersection = float(np.sum(p * t))
    union = float(np.sum(np.clip(p + t, 0, 1)))
    p_sum = float(np.sum(p))
    t_sum = float(np.sum(t))

    dice = (2.0 * intersection + eps) / (p_sum + t_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou


def make_overlay_rgb(mri_rgb: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Creates an RGB diagnostic overlay on top of grayscale MRI:
      - Green  : True Positive (Correct tumor detection)
      - Red    : False Positive (Over-segmentation / hallucination)
      - Yellow : False Negative (Missed tumor region)
    """
    # Base grayscale image replicated to RGB
    overlay = mri_rgb.copy().astype(np.float32)
    if overlay.max() > 1.0:
        overlay = overlay / 255.0

    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)

    tp = p & g
    fp = p & (~g)
    fn = (~p) & g

    # Blend colors
    alpha = 0.6
    # TP -> Green [0.0, 1.0, 0.0]
    overlay[tp] = (1.0 - alpha) * overlay[tp] + alpha * np.array([0.0, 0.9, 0.2])
    # FP -> Red [1.0, 0.1, 0.1]
    overlay[fp] = (1.0 - alpha) * overlay[fp] + alpha * np.array([1.0, 0.2, 0.2])
    # FN -> Yellow [1.0, 0.85, 0.0]
    overlay[fn] = (1.0 - alpha) * overlay[fn] + alpha * np.array([1.0, 0.85, 0.0])

    return np.clip(overlay, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate & visualize 10 random test samples with U-Net")
    parser.add_argument("--test_images_dir", type=str, default="data/raw/segmentation_task/test/images",
                        help="Path to test images folder")
    parser.add_argument("--test_masks_dir", type=str, default="data/raw/segmentation_task/test/masks",
                        help="Path to test masks folder")
    parser.add_argument("--weights_path", type=str, default="models/segmentation/unet_brisc.pth",
                        help="Path to trained U-Net checkpoint (.pth)")
    parser.add_argument("--output_image", type=str, default="metrics/segmentation/sample_test_comparison_10.png",
                        help="Where to save the output comparison image")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed (default: None for truly random selection every run)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device (cuda or cpu)")
    args = parser.parse_args()

    # Set seed only if explicitly provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    images_dir = Path(args.test_images_dir)
    masks_dir = Path(args.test_masks_dir)
    weights_path = Path(args.weights_path)
    output_path = Path(args.output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Missing images ({images_dir}) or masks ({masks_dir}) directory!")

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    # 1. Match paired test samples
    valid_exts = {".jpg", ".jpeg", ".png"}
    all_images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts])

    pairs: list[tuple[Path, Path]] = []
    for img_p in all_images:
        mask_candidate = masks_dir / (img_p.stem + ".png")
        if not mask_candidate.exists():
            mask_candidate = masks_dir / (img_p.name)
        if mask_candidate.exists():
            pairs.append((img_p, mask_candidate))

    if len(pairs) == 0:
        raise RuntimeError(f"No matching image-mask pairs found in {images_dir} and {masks_dir}")

    print(f"Discovered {len(pairs)} test image-mask pairs.")
    num_to_pick = min(args.num_samples, len(pairs))
    selected_pairs = random.sample(pairs, num_to_pick)

    # 2. Load U-Net Model
    device = torch.device(args.device)
    print(f"Loading U-Net on {device} from: {weights_path}")
    model = UNet(in_channels=3, out_channels=1, base_features=32)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Preprocessing transforms (256x256 matching U-Net)
    img_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    results = []

    # 3. Setup Plot Figure (num_samples rows x 5 columns)
    fig, axes = plt.subplots(num_to_pick, 5, figsize=(18, 3.6 * num_to_pick))
    if num_to_pick == 1:
        axes = np.expand_dims(axes, 0)

    print("\n" + "=" * 80)
    print(f"{'#':<3} | {'Image Filename':<32} | {'Dice (DSC)':<12} | {'IoU':<12} | {'Rating'}")
    print("-" * 80)

    dices, ious = [], []

    for i, (img_path, mask_path) in enumerate(selected_pairs):
        pil_img = Image.open(img_path).convert("RGB")
        pil_mask = Image.open(mask_path).convert("L")

        img_tensor = img_transform(pil_img).unsqueeze(0).to(device)
        gt_tensor = mask_transform(pil_mask).squeeze()
        gt_bin = (gt_tensor > 0.5).cpu().numpy().astype(np.float32)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        pred_bin = (probs > 0.5).astype(np.float32)

        dice, iou = compute_slice_metrics(pred_bin, gt_bin)
        dices.append(dice)
        ious.append(iou)

        rating = "Excellent" if dice >= 0.85 else ("Good" if dice >= 0.70 else "Fair/Poor")
        print(f"{i+1:2d}  | {img_path.name:<32} | {dice*100:6.2f}%     | {iou*100:6.2f}%     | {rating}")

        # Render rows
        mri_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

        # Col 1: Raw MRI
        axes[i, 0].imshow(mri_np)
        axes[i, 0].set_title(f"Sample {i+1}: Raw MRI\n{img_path.stem[:22]}", fontsize=10)
        axes[i, 0].axis("off")

        # Col 2: Ground Truth Mask
        axes[i, 1].imshow(gt_bin, cmap="gray")
        axes[i, 1].set_title(f"Ground Truth\nTumor: {int(np.sum(gt_bin))} px", fontsize=10)
        axes[i, 1].axis("off")

        # Col 3: Predicted Mask
        axes[i, 2].imshow(pred_bin, cmap="gray")
        axes[i, 2].set_title(f"U-Net Prediction\nDice: {dice*100:.1f}% | IoU: {iou*100:.1f}%", fontsize=10)
        axes[i, 2].axis("off")

        # Col 4: Probability Map
        im_prob = axes[i, 3].imshow(probs, cmap="magma", vmin=0.0, vmax=1.0)
        axes[i, 3].set_title("Probability Map", fontsize=10)
        axes[i, 3].axis("off")

        # Col 5: Color Overlay Diagnostic
        overlay = make_overlay_rgb(mri_np, pred_bin, gt_bin)
        axes[i, 4].imshow(overlay)
        axes[i, 4].set_title("Overlay (G:TP, R:FP, Y:FN)", fontsize=10)
        axes[i, 4].axis("off")

    print("-" * 80)
    print(f"Mean Sample Dice: {np.mean(dices)*100:.2f}%  |  Mean Sample IoU: {np.mean(ious)*100:.2f}%")
    print("=" * 80)

    plt.suptitle("U-Net Segmentation: 10 Random Test Slices vs Ground Truth", fontsize=16, y=0.995, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[Success] Visual comparison grid saved to: {output_path}")


if __name__ == "__main__":
    main()
