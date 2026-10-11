"""
Tests for UNetPlusPlus (modules/unet_plusplus.py)

Validates:
  1. UNetPlusPlus forward pass shapes with and without deep supervision.
  2. Eval mode output behavior.
  3. Parameter count comparison against Standard UNet and Attention UNet.
  4. Backward pass gradient flow through all nested dense pathways.
  5. build_unet_plusplus factory function.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from modules.unet import UNet, AttentionUNet, count_parameters as count_unet_params
from modules.unet_plusplus import UNetPlusPlus, build_unet_plusplus, count_parameters


def test_unet_plusplus_shapes_and_modes():
    print("\n--- Test 1: UNet++ Forward Pass Shapes & Modes ---")
    B, C, H, W = 2, 3, 256, 256
    dummy = torch.randn(B, C, H, W)

    # 1. With deep supervision in training mode
    model_ds = UNetPlusPlus(in_channels=3, out_channels=1, base_features=32, deep_supervision=True)
    model_ds.train()
    outputs = model_ds(dummy)
    assert isinstance(outputs, list), "Expected list of outputs in training mode with deep_supervision=True"
    assert len(outputs) == 4, f"Expected 4 deep supervision heads, got {len(outputs)}"
    for idx, out in enumerate(outputs, 1):
        assert out.shape == (B, 1, H, W), f"Head {idx} shape mismatch: {out.shape}"
    print(f"  [PASS] Training deep supervision: 4 heads of shape {tuple(outputs[0].shape)}")

    # 2. With deep supervision in eval mode
    model_ds.eval()
    with torch.no_grad():
        eval_out = model_ds(dummy)
    assert isinstance(eval_out, torch.Tensor), "Expected single tensor in eval mode"
    assert eval_out.shape == (B, 1, H, W), f"Eval output shape mismatch: {eval_out.shape}"
    print(f"  [PASS] Eval mode output: single tensor of shape {tuple(eval_out.shape)}")

    # 3. Without deep supervision
    model_no_ds = UNetPlusPlus(in_channels=3, out_channels=1, base_features=32, deep_supervision=False)
    model_no_ds.train()
    no_ds_out = model_no_ds(dummy)
    assert isinstance(no_ds_out, torch.Tensor), "Expected single tensor when deep_supervision=False"
    assert no_ds_out.shape == (B, 1, H, W), f"Shape mismatch: {no_ds_out.shape}"
    print(f"  [PASS] Single head mode: {tuple(no_ds_out.shape)}")


def test_parameter_counts():
    print("\n--- Test 2: Parameter Count Comparison ---")
    base = 32
    std_unet = UNet(in_channels=3, out_channels=1, base_features=base)
    attn_unet = AttentionUNet(in_channels=3, out_channels=1, base_features=base)
    nested_unet = UNetPlusPlus(in_channels=3, out_channels=1, base_features=base, deep_supervision=True)

    p_std = count_unet_params(std_unet)
    p_attn = count_unet_params(attn_unet)
    p_nested = count_parameters(nested_unet)

    print(f"  Standard U-Net:   {p_std:>10,} ({p_std/1e6:.2f}M)")
    print(f"  Attention U-Net:  {p_attn:>10,} ({p_attn/1e6:.2f}M)")
    print(f"  U-Net++ (Nested): {p_nested:>10,} ({p_nested/1e6:.2f}M)")

    # UNet++ adds dense skip intermediate nodes (approx 9M params)
    assert p_nested > p_attn > p_std, "Parameter hierarchy must hold: UNet++ > Attention U-Net > Standard U-Net"
    assert 8.5e6 < p_nested < 9.5e6, f"Expected ~9.04M params for base 32, got {p_nested:,}"
    print(f"  [PASS] Architecture matches paper specifications: {p_nested:,} params.")


def test_gradient_flow_and_backward():
    print("\n--- Test 3: Backward Pass & Gradient Flow ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlus(in_channels=3, out_channels=1, base_features=32, deep_supervision=True).to(device)
    model.train()

    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()
    outputs = model(dummy_input)
    assert isinstance(outputs, list)

    # Average loss across all 4 deep supervision heads
    loss = sum(criterion(out, target) for out in outputs) / len(outputs)
    loss.backward()

    # Check gradients in representative nodes across the grid:
    # Encoder (conv0_0, conv4_0), Intermediate (conv0_1, conv1_2, conv0_4)
    nodes_to_check = [
        ("conv0_0", model.conv0_0),
        ("conv4_0", model.conv4_0),
        ("conv0_1", model.conv0_1),
        ("conv1_2", model.conv1_2),
        ("conv0_4", model.conv0_4),
        ("final1",  model.final1),
        ("final4",  model.final4),
    ]

    for name, module in nodes_to_check:
        for p_name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None in {name}.{p_name}"
                assert torch.norm(param.grad) > 0, f"Gradient is zero in {name}.{p_name}"

    print("  [PASS] Healthy non-zero gradients across encoder, dense skip nodes, and supervision heads.")


def test_factory_function():
    print("\n--- Test 4: Factory Function ---")
    cfg = {
        "in_channels": 3,
        "out_channels": 1,
        "base_features": 32,
        "deep_supervision": True,
    }
    model = build_unet_plusplus(cfg)
    assert isinstance(model, UNetPlusPlus)
    assert model.deep_supervision is True
    print("  [PASS] build_unet_plusplus() instantiated correctly.")


if __name__ == "__main__":
    test_unet_plusplus_shapes_and_modes()
    test_parameter_counts()
    test_gradient_flow_and_backward()
    test_factory_function()
    print("\n================ ALL U-NET++ TESTS PASSED ================\n")
