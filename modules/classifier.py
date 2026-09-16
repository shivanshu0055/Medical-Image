"""
MedBoard — modules/classifier.py
Phase 3: Tumor Classification Model Architecture

Builds an EfficientNet-B0 transfer learning model using `timm` for 4-class
brain MRI classification:
    - glioma
    - meningioma
    - pituitary
    - no_tumor

Why EfficientNet-B0?
  - Pretrained on ImageNet (~4.01M parameters)
  - Highly parameter-efficient and accurate
  - Lightweight memory footprint (~400-600MB VRAM) suitable for RTX 2050 and Colab T4
  - Standard resolution: 224x224 RGB
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CLASSES: list[str] = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]


def build_classifier(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    in_chans: int = 3,
) -> nn.Module:
    """
    Build and return an EfficientNet-B0 model configured for brain MRI classification.

    Args:
        num_classes: Number of output categories (default: 4).
        pretrained: Whether to load ImageNet pretrained weights (default: True).
        dropout_rate: Dropout probability in the classification head (default: 0.2).
        in_chans: Number of input channels (default: 3).

    Returns:
        PyTorch nn.Module (EfficientNet-B0).
    """
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_rate,
        in_chans=in_chans,
    )
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count total and trainable parameters of the model.

    Returns:
        dict with 'total' and 'trainable' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


@torch.no_grad()
def predict_class(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device | str = "cpu",
    class_names: Optional[Sequence[str]] = None,
) -> tuple[str, float, dict[str, float]]:
    """
    Run forward inference on a preprocessed MRI tensor to classify the tumor.

    Args:
        model: Trained classifier model.
        input_tensor: Preprocessed tensor of shape (3, 224, 224) or (1, 3, 224, 224).
        device: Device to run inference on ("cuda" or "cpu").
        class_names: List of class names matching the model's output index order.

    Returns:
        tuple of:
          - predicted_class: name of the top-predicted class (str)
          - confidence: probability of the top class (float, 0.0 to 1.0)
          - class_probabilities: dict mapping each class name to its probability
    """
    if class_names is None:
        class_names = DEFAULT_CLASSES

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Ensure batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    elif input_tensor.ndim != 4:
        raise ValueError(
            f"Expected input_tensor with 3 or 4 dimensions, got shape {input_tensor.shape}"
        )

    input_tensor = input_tensor.to(device, dtype=torch.float32)

    logits = model(input_tensor)
    probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(np.argmax(probabilities))
    predicted_class = class_names[top_idx]
    confidence = float(probabilities[top_idx])

    class_probabilities = {
        name: float(prob) for name, prob in zip(class_names, probabilities)
    }

    return predicted_class, confidence, class_probabilities
