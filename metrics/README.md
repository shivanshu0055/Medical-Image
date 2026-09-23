# MedBoard — Model Evaluation Metrics & Benchmarks

This directory houses the structured evaluation results, performance metrics, confusion matrices, and sample predictions for all models trained and deployed within MedBoard.

---

## 📁 Directory Structure

```
metrics/
├── segmentation/                 # U-Net Segmentation Agent Metrics
│   ├── metrics.json              # Machine-readable evaluation metrics (Dice, IoU, Loss)
│   ├── evaluation_report.md      # Detailed benchmark report
│   └── sample_segmentations.png  # Ground truth vs predicted mask visual comparisons
│
└── classification/               # EfficientNet-B0 Classifier Metrics
    ├── metrics.json              # Machine-readable metrics (Accuracy, Macro-F1, Loss)
    ├── evaluation_report.md      # Full classification report & per-class breakdown
    ├── classification_report.txt # Raw precision, recall, and F1 table
    └── confusion_matrix.png      # 4-class confusion matrix heatmap
```

---

## 🏆 Current Model Performance Summary

| Agent | Architecture | Primary Metric | Score | Rating |
|---|---|---|---|---|
| **Segmentation Agent** | U-Net (7.85M params) | Test Dice (DSC) | **0.8706** (87.06%) | **Excellent (>0.85)** |
| **Segmentation Agent** | U-Net (7.85M params) | Test IoU (Jaccard) | **0.7709** (77.09%) | High Spatial Overlap |
| **Classification Agent**| EfficientNet-B0 (4.01M params) | Test Accuracy | **98.20%** | State of the Art |
| **Classification Agent**| EfficientNet-B0 (4.01M params) | Test Macro-F1 | **0.9810** | Balanced across classes |
