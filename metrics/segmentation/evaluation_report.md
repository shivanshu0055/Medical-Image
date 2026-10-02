# Segmentation Model Evaluation Report

**Model Architecture**: 2D U-Net (Encoder-Decoder with Skip Connections)  
**Total Parameters**: 7,849,601  
**Checkpoint Path**: `models\segmentation\unet_brisc.pth` (30.0 MB)  
**Dataset**: BRISC 2025 Test Split (860 test slices)  

---

## Benchmark Performance

| Metric | Score | Clinical Standard | Rating |
|---|---|---|---|
| **Dice Similarity Coefficient (DSC)** | **0.8706** (87.06%) | > 0.85 | **Excellent (>0.85)** |
| **Intersection over Union (IoU)** | **0.8026** (80.26%) | > 0.75 | **High Spatial Overlap** |
| **Combined Loss (Dice + BCE)** | **0.0755** | < 0.15 | **Excellent Convergence** |

---

## Visual Verification
![Sample Segmentations](sample_segmentations.png)
