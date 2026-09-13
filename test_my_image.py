"""
MedBoard — Quick Segmentation Test Script

Usage:
    python test_my_image.py <path_to_mri_image>

Example:
    python test_my_image.py data/raw/segmentation_task/test/images/brisc2025_test_00001_gl_ax_t1.jpg

What it does:
    1. Loads your trained U-Net weights
    2. Preprocesses your image
    3. Predicts the tumor mask
    4. Shows: Original | Predicted Mask | Overlay (side by side)
    5. Saves the result as 'seg_result.png'
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from modules.unet import build_unet, predict_mask
from modules.preprocessing import MRIPreprocessor

# ── Config ───────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "models/segmentation/unet_brisc.pth"
THRESHOLD    = 0.5
OUTPUT_PATH  = "seg_result.png"

# ── Load image path from command line ────────────────────────────────────────
if len(sys.argv) < 2:
    # Default to a random test image if no argument given
    test_dir = Path("data/raw/segmentation_task/test/images")
    images   = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not images:
        print("Usage: python test_my_image.py <path_to_image>")
        sys.exit(1)
    image_path = images[0]
    print(f"No image specified. Using: {image_path.name}")
else:
    image_path = Path(sys.argv[1])

if not image_path.exists():
    print(f"Image not found: {image_path}")
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model from {WEIGHTS_PATH}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = build_unet()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded.")

# ── Preprocess image ──────────────────────────────────────────────────────────
print(f"\nProcessing: {image_path.name}")
preprocessor = MRIPreprocessor()
preprocessed = preprocessor.preprocess_image(image_path, mode="segmentation")

# ── Predict mask ──────────────────────────────────────────────────────────────
mask_tensor = predict_mask(model, preprocessed.tensor, threshold=THRESHOLD, device=str(device))
# mask_tensor shape: (1, 256, 256), values: {0.0, 1.0}

# ── Convert to numpy for visualization ────────────────────────────────────────
# Original image: resize to 256x256 for fair comparison
original_img = Image.open(image_path).convert("RGB").resize((256, 256))
original_np  = np.array(original_img)

# Mask: (1, 256, 256) → (256, 256)
mask_np = mask_tensor.squeeze(0).cpu().numpy()

# Overlay: original + red tumor region
overlay_np = original_np.copy()
tumor_region = mask_np > 0.5
overlay_np[tumor_region, 0] = 255   # Red channel max
overlay_np[tumor_region, 1] = (overlay_np[tumor_region, 1] * 0.3).astype(np.uint8)
overlay_np[tumor_region, 2] = (overlay_np[tumor_region, 2] * 0.3).astype(np.uint8)

# ── Calculate tumor stats ─────────────────────────────────────────────────────
tumor_pixels  = mask_np.sum()
total_pixels  = mask_np.size
tumor_pct     = 100 * tumor_pixels / total_pixels
has_tumor     = tumor_pct > 0.1

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#1a1a2e")

titles = ["Original MRI", "Predicted Tumor Mask", "Overlay"]
images_to_show = [original_np, mask_np, overlay_np]
cmaps  = [None, "Reds", None]

for ax, title, img, cmap in zip(axes, titles, images_to_show, cmaps):
    ax.set_facecolor("#0f0f23")
    if cmap:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(img)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

# Status box
status_color = "#ff4444" if has_tumor else "#44ff88"
status_text  = f"TUMOR DETECTED\n{tumor_pct:.1f}% of image" if has_tumor else "NO TUMOR DETECTED"
fig.text(0.5, 0.02, status_text, ha="center", va="bottom",
         fontsize=14, fontweight="bold", color=status_color,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f0f23", edgecolor=status_color, linewidth=2))

fig.suptitle(f"MedBoard — Segmentation Result\n{image_path.name}",
             color="white", fontsize=14, y=1.0)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nResult saved to: {OUTPUT_PATH}")

# ── Print summary ──────────────────────────────────────────────────────────────
print("\n" + "="*45)
print("  SEGMENTATION RESULT")
print("="*45)
print(f"  Image:        {image_path.name}")
print(f"  Tumor found:  {'YES' if has_tumor else 'NO'}")
print(f"  Tumor area:   {tumor_pct:.2f}% of image")
print(f"  Tumor pixels: {int(tumor_pixels)} / {total_pixels}")
print("="*45)

plt.show()
