"""
MedBoard — tests/test_classification.py
Unit tests for Phase 3 Classification components:
  1. EfficientNet-B0 architecture builder & parameter count
  2. Inference prediction function
  3. BRISC dataset scanning & DataLoader creation
  4. Training step & gradient flow verification
  5. LangGraph classification agent integration
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from modules.classifier import (
    DEFAULT_CLASSES,
    build_classifier,
    count_parameters,
    predict_class,
)
from modules.clf_dataset import (
    create_classification_dataloaders,
    scan_dataset_dir,
)
from modules.clf_trainer import train_one_epoch
from agents.classification_agent import ClassificationAgent
from pipeline import MedBoardState


def test_model_architecture():
    """Verify EfficientNet-B0 builds with correct parameter count and output shape."""
    model = build_classifier(num_classes=4, pretrained=False)
    params = count_parameters(model)

    assert 3_900_000 < params["total"] < 4_200_000, (
        f"Expected ~4.01M parameters, got {params['total']:,}"
    )

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"


def test_predict_class_helper():
    """Verify predict_class returns valid class, confidence, and probability distribution."""
    model = build_classifier(num_classes=4, pretrained=False)
    dummy_input = torch.randn(3, 224, 224)

    pred_class, confidence, probs = predict_class(
        model=model,
        input_tensor=dummy_input,
        device="cpu",
        class_names=DEFAULT_CLASSES,
    )

    assert pred_class in DEFAULT_CLASSES, f"Unknown class: {pred_class}"
    assert 0.0 <= confidence <= 1.0, f"Invalid confidence score: {confidence}"
    assert set(probs.keys()) == set(DEFAULT_CLASSES)
    assert pytest.approx(sum(probs.values()), rel=1e-3) == 1.0


def test_classification_dataloaders():
    """Verify dataset scanner and DataLoaders on real BRISC data if available."""
    train_dir = Path("data/raw/classification_task/train")
    test_dir = Path("data/raw/classification_task/test")

    if not train_dir.exists():
        pytest.skip("BRISC classification dataset not found locally.")

    train_loader, val_loader, test_loader = create_classification_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        val_ratio=0.15,
        batch_size=4,
        image_size=224,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 224, 224), f"Unexpected image batch shape: {images.shape}"
    assert labels.shape == (4,), f"Unexpected label batch shape: {labels.shape}"
    assert labels.dtype == torch.int64
    assert all(0 <= int(lbl) < 4 for lbl in labels)


def test_training_step_gradient_flow():
    """Verify forward and backward pass compute valid gradients."""
    model = build_classifier(num_classes=4, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    dummy_loader = [(dummy_images, dummy_labels)]

    loss, acc = train_one_epoch(
        model=model,
        loader=dummy_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert loss > 0.0
    assert 0.0 <= acc <= 1.0

    # Ensure gradients were computed on parameters
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert has_grad, "No valid gradients found after optimizer step."


def test_classification_agent_graceful_missing_weights():
    """Verify ClassificationAgent reports descriptive error when weights are not yet trained."""
    agent = ClassificationAgent()
    agent.weights_path = "models/classification/non_existent_weights.pth"

    state = MedBoardState(image_path="dummy.jpg")
    updated_state = agent.run(state)

    assert len(updated_state.errors) > 0
    assert any("Classification model weights not found" in err for err in updated_state.errors)


def test_classification_agent_real_inference():
    """Verify ClassificationAgent runs end-to-end inference with real trained weights."""
    weights_path = Path("models/classification/efficientnet_b0_brisc.pth")
    if not weights_path.exists():
        pytest.skip("Classification weights not found.")

    test_image = next(Path("data/raw/classification_task/test/glioma").glob("*.jpg"), None)
    if test_image is None:
        pytest.skip("Test image not found.")

    agent = ClassificationAgent()
    state = MedBoardState(image_path=str(test_image))
    updated_state = agent.run(state)

    assert len(updated_state.errors) == 0, f"Inference errors: {updated_state.errors}"
    assert updated_state.predicted_class in DEFAULT_CLASSES
    assert 0.0 <= updated_state.classification_confidence <= 1.0
    assert len(updated_state.class_probabilities) == 4
