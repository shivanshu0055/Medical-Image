"""
Tests for AttentionGate and AttentionUNet (modules/unet.py)

Validates:
  1. AttentionGate forward pass & attention map range [0, 1].
  2. AttentionDecoderBlock shape consistency.
  3. AttentionUNet forward pass (B, 3, 256, 256) -> (B, 1, 256, 256).
  4. Parameter count comparison against standard UNet.
  5. Backward pass gradient flow through attention gates.
  6. build_unet and build_attention_unet factory functions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from modules.unet import (
    AttentionGate,
    AttentionDecoderBlock,
    AttentionUNet,
    UNet,
    build_attention_unet,
    build_unet,
    count_parameters,
)


def test_attention_gate_shapes_and_values():
    print("\n--- Test 1: AttentionGate Shapes & Values ---")
    gate_channels = 256
    skip_channels = 128
    inter_channels = 64
    B, H, W = 2, 64, 64

    ag = AttentionGate(gate_channels=gate_channels, skip_channels=skip_channels, inter_channels=inter_channels)
    g = torch.randn(B, gate_channels, H // 2, W // 2)  # coarse gating signal (32x32)
    x = torch.randn(B, skip_channels, H, W)             # fine skip features (64x64)

    out = ag(g, x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert hasattr(ag, "last_alpha"), "AttentionGate should cache last_alpha"
    alpha = ag.last_alpha
    assert alpha.shape == (B, 1, H, W), f"Expected alpha shape {(B, 1, H, W)}, got {alpha.shape}"
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all(), "Attention weights must be within [0, 1]"
    print(f"  [PASS] AttentionGate output: {tuple(out.shape)}, alpha range: [{alpha.min().item():.3f}, {alpha.max().item():.3f}]")


def test_attention_decoder_block():
    print("\n--- Test 2: AttentionDecoderBlock Forward Pass ---")
    gate_ch = 512
    skip_ch = 256
    out_ch = 256
    B, H, W = 2, 16, 16

    dec_block = AttentionDecoderBlock(gate_channels=gate_ch, skip_channels=skip_ch, out_channels=out_ch)
    x = torch.randn(B, gate_ch, H, W)       # e.g. from bottleneck (16x16)
    skip = torch.randn(B, skip_ch, H * 2, W * 2)  # skip connection from encoder (32x32)

    out = dec_block(x, skip)
    assert out.shape == (B, out_ch, H * 2, W * 2), f"Expected {(B, out_ch, H * 2, W * 2)}, got {out.shape}"
    print(f"  [PASS] AttentionDecoderBlock output: {tuple(out.shape)}")


def test_attention_unet_forward_and_params():
    print("\n--- Test 3: AttentionUNet Parameters & Forward Pass ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    std_unet = UNet(in_channels=3, out_channels=1, base_features=32).to(device)
    attn_unet = AttentionUNet(in_channels=3, out_channels=1, base_features=32).to(device)

    std_params = count_parameters(std_unet)
    attn_params = count_parameters(attn_unet)
    diff_params = attn_params - std_params
    pct_increase = (diff_params / std_params) * 100

    print(f"  Standard UNet params:    {std_params:>10,}")
    print(f"  Attention UNet params:   {attn_params:>10,}")
    print(f"  Added AG params:         {diff_params:>10,} (+{pct_increase:.2f}%)")

    # Shape flow
    dummy = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        out_std = std_unet(dummy)
        out_attn = attn_unet(dummy)

    assert out_attn.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out_attn.shape}"
    assert out_std.shape == out_attn.shape, "Standard and Attention UNet output shapes must match"
    print(f"  [PASS] Forward output shape: {tuple(out_attn.shape)}")


def test_attention_unet_gradient_flow():
    print("\n--- Test 4: Backward Pass & Gradient Flow ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=3, out_channels=1, base_features=32).to(device)
    model.train()

    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()
    logits = model(dummy_input)
    loss = criterion(logits, target)
    loss.backward()

    # Check gradients in each attention gate
    for name, dec in [("dec4", model.dec4), ("dec3", model.dec3), ("dec2", model.dec2), ("dec1", model.dec1)]:
        wg_grad = dec.attn.W_g[0].weight.grad
        wx_grad = dec.attn.W_x[0].weight.grad
        psi_grad = dec.attn.psi[0].weight.grad

        assert wg_grad is not None and torch.norm(wg_grad) > 0, f"Gradient vanished in {name} W_g"
        assert wx_grad is not None and torch.norm(wx_grad) > 0, f"Gradient vanished in {name} W_x"
        assert psi_grad is not None and torch.norm(psi_grad) > 0, f"Gradient vanished in {name} psi"

    print("  [PASS] Non-zero gradients confirmed across all Attention Gates.")


def test_factory_functions():
    print("\n--- Test 5: Factory Functions & Config Routing ---")
    cfg_std = {"architecture": "unet", "base_features": 32}
    cfg_attn = {"architecture": "attention_unet", "base_features": 32}

    m1 = build_unet(cfg_std)
    m2 = build_unet(cfg_attn)
    m3 = build_attention_unet(cfg_attn)

    assert isinstance(m1, UNet), f"Expected UNet, got {type(m1)}"
    assert isinstance(m2, AttentionUNet), f"Expected AttentionUNet, got {type(m2)}"
    assert isinstance(m3, AttentionUNet), f"Expected AttentionUNet, got {type(m3)}"
    print("  [PASS] Factory functions route correctly.")


if __name__ == "__main__":
    test_attention_gate_shapes_and_values()
    test_attention_decoder_block()
    test_attention_unet_forward_and_params()
    test_attention_unet_gradient_flow()
    test_factory_functions()
    print("\n================ ALL TESTS PASSED ================\n")
