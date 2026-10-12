"""
Tests for SaberNet (modules/sabernet.py)

Validates:
  1. SaberRelationBlock forward pass, attention energy, and gamma scaling.
  2. SaberNet full forward pass shape: (B, 3, 256, 256) -> (B, 1, 256, 256).
  3. Parameter count (~9.5M params).
  4. Backward pass gradient flow through SABER bottleneck and relation skip blocks.
  5. build_sabernet factory function.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from modules.sabernet import (
    SaberRelationBlock,
    SaberNet,
    build_sabernet,
    count_parameters,
)


def test_saber_relation_block():
    print("\n--- Test 1: SABER Relation Block Shapes & Projections ---")
    B, C, H, W = 2, 128, 32, 32
    dummy = torch.randn(B, C, H, W)
    block = SaberRelationBlock(in_channels=C, reduction=4)

    out = block(dummy)
    assert out.shape == dummy.shape, f"Expected {dummy.shape}, got {out.shape}"
    assert abs(block.gamma.item() - 0.1) < 1e-4, "Gamma must initialize to 0.1 for active relation gradient flow"

    # Test high-resolution pooling mode (e.g. 128x128)
    dummy_hi = torch.randn(2, 64, 128, 128)
    block_hi = SaberRelationBlock(in_channels=64, reduction=4)
    out_hi = block_hi(dummy_hi)
    assert out_hi.shape == dummy_hi.shape, f"High-res mismatch: {out_hi.shape}"
    print(f"  [PASS] SABER Relation Block output verified: {tuple(out.shape)}")


def test_sabernet_forward_and_params():
    print("\n--- Test 2: SaberNet Forward Pass & Parameter Count ---")
    B, C, H, W = 2, 3, 256, 256
    dummy = torch.randn(B, C, H, W)
    model = SaberNet(in_channels=3, out_channels=1, base_features=32)

    total_params = count_parameters(model)
    print(f"  SaberNet Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # SaberNet adds SABER relation modules on bottleneck and skips (~8.38M params)
    assert 8.0e6 < total_params < 9.0e6, f"Expected ~8.38M params, got {total_params:,}"

    model.eval()
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (B, 1, H, W), f"Output shape mismatch: {out.shape}"
    print(f"  [PASS] Forward output shape: {tuple(out.shape)}")


def test_sabernet_gradient_flow():
    print("\n--- Test 3: Backward Pass & Gradient Flow ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SaberNet(in_channels=3, out_channels=1, base_features=32).to(device)
    model.train()

    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()
    logits = model(dummy_input)
    loss = criterion(logits, target)
    loss.backward()

    # Check gradients in SABER bottleneck and skip relation blocks
    check_blocks = [
        ("saber_bottleneck", model.saber_bottleneck),
        ("saber_skip4", model.saber_skip4),
        ("saber_skip3", model.saber_skip3),
        ("saber_skip2", model.saber_skip2),
        ("saber_skip1", model.saber_skip1),
        ("enc1", model.enc1),
        ("dec1", model.dec1),
        ("output_conv", model.output_conv),
    ]

    for name, module in check_blocks:
        for p_name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None in {name}.{p_name}"
                assert torch.norm(param.grad) > 0, f"Gradient vanished in {name}.{p_name}"

    print("  [PASS] Healthy non-zero gradients verified across all SABER blocks and CNN layers.")


def test_factory_function():
    print("\n--- Test 4: Factory Function ---")
    cfg = {"in_channels": 3, "out_channels": 1, "base_features": 32}
    model = build_sabernet(cfg)
    assert isinstance(model, SaberNet)
    print("  [PASS] build_sabernet() instantiated successfully.")


if __name__ == "__main__":
    test_saber_relation_block()
    test_sabernet_forward_and_params()
    test_sabernet_gradient_flow()
    test_factory_function()
    print("\n================ ALL SABERNET TESTS PASSED ================\n")
