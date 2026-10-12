"""
Tests for Pretrained ResNet34 U-Net++ (modules/unet_plusplus_resnet.py)

Validates:
  1. Forward pass shapes with and without deep supervision.
  2. Eval mode output behavior.
  3. Pretrained parameter count (~24.4M).
  4. Differential learning rate parameter groups.
  5. Backward pass gradient flow through both pretrained encoder and new decoder.
  6. build_resnet34_unet_plusplus factory function.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from modules.unet_plusplus_resnet import (
    ResNet34UNetPlusPlus,
    build_resnet34_unet_plusplus,
    count_parameters,
)


def test_resnet34_unet_plusplus_shapes():
    print("\n--- Test 1: ResNet-34 U-Net++ Forward Shapes ---")
    B, C, H, W = 2, 3, 256, 256
    dummy = torch.randn(B, C, H, W)

    # 1. Training mode with deep supervision (returns 4 heads)
    # Using pretrained=False for quick unit testing (avoids download in test)
    model = ResNet34UNetPlusPlus(in_channels=3, out_channels=1, pretrained=False, deep_supervision=True)
    model.train()
    outputs = model(dummy)
    assert isinstance(outputs, list), "Expected list of outputs in training mode with deep_supervision=True"
    assert len(outputs) == 4, f"Expected 4 heads, got {len(outputs)}"
    for idx, out in enumerate(outputs, 1):
        assert out.shape == (B, 1, H, W), f"Head {idx} shape mismatch: {out.shape}"
    print(f"  [PASS] Training deep supervision: 4 heads of shape {tuple(outputs[0].shape)}")

    # 2. Eval mode (returns single tensor from X_0,4)
    model.eval()
    with torch.no_grad():
        eval_out = model(dummy)
    assert isinstance(eval_out, torch.Tensor), "Expected single tensor in eval mode"
    assert eval_out.shape == (B, 1, H, W), f"Eval output shape mismatch: {eval_out.shape}"
    print(f"  [PASS] Eval mode output: single tensor of shape {tuple(eval_out.shape)}")


def test_parameter_groups_and_count():
    print("\n--- Test 2: Parameter Groups & Count ---")
    model = ResNet34UNetPlusPlus(in_channels=3, out_channels=1, pretrained=False, deep_supervision=True)
    total_params = count_parameters(model)
    print(f"  Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Verify ResNet-34 based model size is ~24M params
    assert 22e6 < total_params < 26e6, f"Expected ~24.4M params, got {total_params:,}"

    # Verify differential parameter groups
    groups = model.get_parameter_groups(encoder_lr=6e-5, decoder_lr=1.8e-4, weight_decay=1e-4)
    assert len(groups) == 2, f"Expected 2 parameter groups, got {len(groups)}"
    assert groups[0]["lr"] == 6e-5, f"Expected encoder LR 6e-5, got {groups[0]['lr']}"
    assert groups[1]["lr"] == 1.8e-4, f"Expected decoder LR 1.8e-4, got {groups[1]['lr']}"

    total_grouped = sum(sum(p.numel() for p in g["params"]) for g in groups)
    assert total_grouped == total_params, f"Mismatch: grouped {total_grouped:,} vs total {total_params:,}"
    print(f"  [PASS] Differential LR groups correctly formed: Encoder (6e-5), Decoder (1.8e-4).")


def test_gradient_flow():
    print("\n--- Test 3: Backward Pass & Gradient Flow ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34UNetPlusPlus(in_channels=3, out_channels=1, pretrained=False, deep_supervision=True).to(device)
    model.train()

    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()
    outputs = model(dummy_input)
    loss = sum(criterion(out, target) for out in outputs) / len(outputs)
    loss.backward()

    # Check gradients in both encoder (resnet conv1, layer4) and decoder (conv0_1, conv0_4, final4)
    check_modules = [
        ("encoder_stem", model.resnet.conv1),
        ("encoder_bottleneck", model.resnet.layer4[0].conv1),
        ("decoder_conv0_1", model.conv0_1),
        ("decoder_conv0_4", model.conv0_4),
        ("output_final4", model.final4),
    ]

    for name, module in check_modules:
        for p_name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None in {name}.{p_name}"
                assert torch.norm(param.grad) > 0, f"Gradient is zero in {name}.{p_name}"

    print("  [PASS] Non-zero gradients verified across pretrained backbone and nested decoder.")


def test_factory_function():
    print("\n--- Test 4: Factory Function ---")
    cfg = {
        "in_channels": 3,
        "out_channels": 1,
        "pretrained": False,
        "deep_supervision": True,
    }
    model = build_resnet34_unet_plusplus(cfg)
    assert isinstance(model, ResNet34UNetPlusPlus)
    print("  [PASS] build_resnet34_unet_plusplus() instantiated correctly.")


if __name__ == "__main__":
    test_resnet34_unet_plusplus_shapes()
    test_parameter_groups_and_count()
    test_gradient_flow()
    test_factory_function()
    print("\n================ ALL RESNET-34 U-NET++ TESTS PASSED ================\n")
