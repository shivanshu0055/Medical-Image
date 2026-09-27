"""
Dry-run test: Verify GeoSampleUNet shape flow, gradient flow, and parameter count.
Compares against the standard UNet for reference.

Uses batch_size=1 for local testing. On Colab T4, use batch_size=16.
"""

import sys
sys.path.insert(0, ".")

import torch
from modules.unet import UNet, count_parameters as count_params_std
from modules.geosample_unet import GeoSampleUNet, count_parameters

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 70)

    # ── Build both models ────────────────────────────────────────────────
    std_unet = UNet(in_channels=3, out_channels=1, base_features=32).to(device)
    geo_unet = GeoSampleUNet(in_channels=3, out_channels=1, base_features=32).to(device)

    # ── Parameter comparison ─────────────────────────────────────────────
    std_params = count_params_std(std_unet)
    geo_params = count_parameters(geo_unet)
    reduction  = (1 - geo_params / std_params) * 100

    print(f"Standard U-Net parameters:   {std_params:>12,}")
    print(f"GeoSample U-Net parameters:  {geo_params:>12,}")
    print(f"Parameter change:            {reduction:>+11.1f}%")
    print("=" * 70)

    # ── Forward pass shape test (batch=1 for local GPU) ──────────────────
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 256, 256, device=device)

    print(f"\n[Forward Pass Test] (batch_size={batch_size})")
    print(f"  Input shape:  {tuple(dummy_input.shape)}")

    with torch.no_grad():
        std_out = std_unet(dummy_input)
        geo_out = geo_unet(dummy_input)

    print(f"  Std U-Net output shape:  {tuple(std_out.shape)}")
    print(f"  Geo U-Net output shape:  {tuple(geo_out.shape)}")

    assert std_out.shape == geo_out.shape, "Shape mismatch!"
    print("  [PASS] Output shapes match.")

    # Free standard model to save VRAM
    del std_unet, std_out
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Backward pass gradient test ──────────────────────────────────────
    print("\n[Backward Pass Test]")
    geo_unet.train()
    dummy_input2 = torch.randn(batch_size, 3, 256, 256, device=device)
    output = geo_unet(dummy_input2)
    loss = output.mean()
    loss.backward()

    # Check that all parameters received gradients
    no_grad_params = []
    for name, param in geo_unet.named_parameters():
        if param.requires_grad and param.grad is None:
            no_grad_params.append(name)

    if no_grad_params:
        print(f"  [WARN] {len(no_grad_params)} parameters got no gradient:")
        for name in no_grad_params[:10]:
            print(f"    - {name}")
    else:
        print(f"  [PASS] All {geo_params:,} parameters received gradients.")

    # ── Memory estimate for Colab T4 ─────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.synchronize()
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        est_b16 = mem_mb * 16 * 0.5  # scale to batch=16 with FP16
        print(f"\n[VRAM Usage]")
        print(f"  Peak (batch={batch_size}, FP32): {mem_mb:.0f} MB")
        print(f"  Estimated (batch=16, FP16):      ~{est_b16:.0f} MB")
        print(f"  T4 headroom (16 GB):             ~{16384 - est_b16:.0f} MB remaining")

    print("\n" + "=" * 70)
    print("[SUCCESS] GeoSampleUNet is ready for training on Colab T4.")
    print("=" * 70)


if __name__ == "__main__":
    main()
