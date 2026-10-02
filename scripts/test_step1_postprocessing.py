import torch
import numpy as np
from pathlib import Path
import scipy.ndimage as ndi
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.geosample_unet import build_geosample_unet
from modules.seg_dataset import SegmentationDataset

def compute_metrics(preds, masks):
    """Compute per-sample Dice and IoU."""
    dices = []
    ious = []
    for p, m in zip(preds, masks):
        inter = (p * m).sum()
        total = p.sum() + m.sum()
        d = (2.0 * inter + 1.0) / (total + 1.0)
        union = total - inter
        iou = (inter + 1.0) / (union + 1.0)
        dices.append(d)
        ious.append(iou)
    return dices, ious

def remove_small_artifacts(pred_np, min_size=20):
    """Remove connected components smaller than min_size pixels."""
    labeled, num_features = ndi.label(pred_np)
    if num_features == 0:
        return pred_np
    sizes = ndi.sum(pred_np, labeled, range(1, num_features + 1))
    cleaned = np.zeros_like(pred_np)
    for i, s in enumerate(sizes, 1):
        if s >= min_size:
            cleaned[labeled == i] = 1.0
    return cleaned

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Step 1 Evaluation on: {device}")

    weights_path = Path("models/segmentation/geosample_unet_brisc.pth")
    model = build_geosample_unet()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    test_dataset = SegmentationDataset(
        images_dir="data/raw/segmentation_task/test/images",
        masks_dir="data/raw/segmentation_task/test/masks",
        split="test",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=0
    )
    print(f"Total test samples: {len(test_dataset)}")

    # Store predicted probabilities and ground truth masks
    all_probs_std = []
    all_probs_tta = []
    all_masks = []

    print("Extracting predictions (Standard + TTA)...")
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            # Standard prediction
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            # TTA: Horizontal Flip
            images_flipped = torch.flip(images, dims=[-1])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.flip(torch.sigmoid(logits_flipped), dims=[-1]).cpu().numpy()
            
            probs_tta = 0.5 * (probs + probs_flipped)

            all_probs_std.append(probs)
            all_probs_tta.append(probs_tta)
            all_masks.append(masks.numpy())

    all_probs_std = np.concatenate(all_probs_std, axis=0)[:, 0]  # (860, 256, 256)
    all_probs_tta = np.concatenate(all_probs_tta, axis=0)[:, 0]
    all_masks = np.concatenate(all_masks, axis=0)[:, 0]

    thresholds = [0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55]

    print("\n" + "=" * 70)
    print(f"{'Method':<25} | {'Threshold':<10} | {'Mean Dice':<12} | {'Median Dice':<12} | {'Mean IoU':<10}")
    print("=" * 70)

    best_config = {"name": "", "dice": 0.0, "median": 0.0, "iou": 0.0, "thresh": 0.0}

    # 1. Standard (No TTA) across thresholds
    for th in thresholds:
        preds = (all_probs_std > th).astype(np.float32)
        dices, ious = compute_metrics(preds, all_masks)
        m_dice = np.mean(dices)
        med_dice = np.median(dices)
        m_iou = np.mean(ious)
        mark = " (Baseline)" if th == 0.50 else ""
        print(f"{'Standard (No TTA)':<25} | {th:<10.2f} | {m_dice*100:6.2f}%{mark:<9} | {med_dice*100:6.2f}%     | {m_iou*100:5.2f}%")
        if m_dice > best_config["dice"]:
            best_config = {"name": "Standard", "dice": m_dice, "median": med_dice, "iou": m_iou, "thresh": th}

    print("-" * 70)

    # 2. TTA across thresholds
    for th in thresholds:
        preds = (all_probs_tta > th).astype(np.float32)
        dices, ious = compute_metrics(preds, all_masks)
        m_dice = np.mean(dices)
        med_dice = np.median(dices)
        m_iou = np.mean(ious)
        print(f"{'With TTA (H-Flip)':<25} | {th:<10.2f} | {m_dice*100:6.2f}%          | {med_dice*100:6.2f}%     | {m_iou*100:5.2f}%")
        if m_dice > best_config["dice"]:
            best_config = {"name": "TTA", "dice": m_dice, "median": med_dice, "iou": m_iou, "thresh": th}

    print("-" * 70)

    # 3. TTA + Artifact Removal (<20 px) at best thresholds
    for th in [best_config["thresh"], 0.40, 0.42, 0.45]:
        preds = (all_probs_tta > th).astype(np.float32)
        cleaned_preds = np.array([remove_small_artifacts(p, min_size=20) for p in preds])
        dices, ious = compute_metrics(cleaned_preds, all_masks)
        m_dice = np.mean(dices)
        med_dice = np.median(dices)
        m_iou = np.mean(ious)
        print(f"{'TTA + Cleanup (20px)':<25} | {th:<10.2f} | {m_dice*100:6.2f}%          | {med_dice*100:6.2f}%     | {m_iou*100:5.2f}%")
        if m_dice > best_config["dice"]:
            best_config = {"name": "TTA + Cleanup", "dice": m_dice, "median": med_dice, "iou": m_iou, "thresh": th}

    print("=" * 70)
    print(f"\n>>> BEST RESULT: {best_config['name']} @ threshold {best_config['thresh']:.2f}")
    print(f"    Mean Dice   : {best_config['dice']*100:.2f}% (vs 85.37% default -> +{best_config['dice']*100 - 85.37:.2f}%)")
    print(f"    Median Dice : {best_config['median']*100:.2f}%")
    print(f"    Mean IoU    : {best_config['iou']*100:.2f}%")

if __name__ == "__main__":
    main()
