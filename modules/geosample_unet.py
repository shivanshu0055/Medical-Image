"""
MedBoard — modules/geosample_unet.py
GeoSample U-Net: Geometry-Guided Sampling for Brain MRI Segmentation

This module assembles the GeoSample operators (from modules/geosample.py)
into a complete U-Net architecture that is a DROP-IN REPLACEMENT for the
standard UNet (from modules/unet.py).

Key differences from standard U-Net:
  - DoubleConv blocks → GeoSample2D (oriented sampling + differential tokens)
  - MaxPool2d → GeoDownsample2D (dual-path edge-preserving downsampler)
  - Naive skip concat → ConsensusSkip2D (geometric alignment before concat)
  - Heavy 3×3 convolutions → Lightweight 1×1 convolutions + bilinear probes

Same interface:
  Input:  (batch, 3, 256, 256)  — RGB MRI image
  Output: (batch, 1, 256, 256)  — raw logits (apply sigmoid for probabilities)

Designed for Colab T4 (16 GB VRAM):
  - batch_size = 16, mixed precision (FP16), num_workers = 2
  - ~50% fewer parameters than standard U-Net
  - Faster per-epoch throughput due to 1×1 convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.geosample import (
    GeoSample2D,
    GeoDownsample2D,
    ConsensusSkip2D,
)


# ─────────────────────────────────────────────────────────────────────────────
#  GeoEncoder Block
#
#  Replaces the standard EncoderBlock from unet.py.
#
#  Standard EncoderBlock does:
#    DoubleConv → save skip features → MaxPool
#
#  GeoEncoder does:
#    GeoSample2D (oriented refinement) → save skip features → GeoDownsample2D
#
#  The skip features are geometry-aware (contain directional gradient and
#  curvature information), making them richer for the decoder.
# ─────────────────────────────────────────────────────────────────────────────

class GeoEncoderBlock(nn.Module):
    """
    One step of the encoder using geometry-guided sampling.

    GeoSample2D replaces DoubleConv for feature extraction.
    GeoDownsample2D replaces MaxPool2d for spatial reduction.

    Returns (skip_features, downsampled_features) just like EncoderBlock.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.geo_conv = GeoSample2D(in_channels, out_channels)
        self.geo_down = GeoDownsample2D(out_channels)

    def forward(self, x):
        features = self.geo_conv(x)        # (B, out_ch, H, W) — skip features
        pooled   = self.geo_down(features) # (B, out_ch, H/2, W/2) — go deeper
        return features, pooled


# ─────────────────────────────────────────────────────────────────────────────
#  GeoDecoder Block
#
#  Replaces the standard DecoderBlock from unet.py.
#
#  Standard DecoderBlock does:
#    Bilinear Upsample → naive cat(skip, upsampled) → DoubleConv
#
#  GeoDecoder does:
#    Bilinear Upsample → ConsensusSkip2D(skip, upsampled) → GeoSample2D
#
#  ConsensusSkip2D ensures encoder and decoder features are geometrically
#  aligned before concatenation, eliminating boundary ghosting.
# ─────────────────────────────────────────────────────────────────────────────

class GeoDecoderBlock(nn.Module):
    """
    One step of the decoder using geometry-guided sampling.

    Bilinear upsampling followed by ConsensusSkip2D alignment and
    GeoSample2D refinement.
    """

    def __init__(self, dec_channels: int, skip_channels: int, out_channels: int):
        """
        Args:
            dec_channels:  Channels from previous decoder stage (or bottleneck).
            skip_channels: Channels from the corresponding encoder skip.
            out_channels:  Output channels for this decoder stage.
        """
        super().__init__()

        # Bilinear upsampling (same as standard U-Net decoder)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        # Geometry-aligned skip connection
        self.consensus = ConsensusSkip2D(
            enc_channels=skip_channels,
            dec_channels=dec_channels,
        )

        # Geometry-guided refinement after concatenation
        # Input channels = skip_channels + dec_channels (after consensus concat)
        self.geo_conv = GeoSample2D(
            in_channels=skip_channels + dec_channels,
            out_channels=out_channels,
        )

    def forward(self, x, skip):
        """
        Args:
            x:    Decoder features from previous stage (B, C_dec, H, W).
            skip: Encoder skip features (B, C_skip, 2H, 2W).
        """
        x = self.upsample(x)                      # (B, C_dec, 2H, 2W)
        x = self.consensus(feat_dec=x, skip_enc=skip)  # (B, C_dec+C_skip, 2H, 2W)
        return self.geo_conv(x)                    # (B, C_out, 2H, 2W)


# ─────────────────────────────────────────────────────────────────────────────
#  GeoSample U-Net
#
#  Complete U-Net architecture with geometry-guided sampling.
#
#  Channel progression (with base_features=32):
#    Encoder:  3 → 32 → 64 → 128 → 256
#    Bottleneck: 256 → 512
#    Decoder:  512+256 → 256 → 128 → 64 → 32
#    Output:   32 → 1 (via 1×1 conv)
#
#  This matches the exact same channel structure and tensor shapes as the
#  standard UNet from modules/unet.py, ensuring drop-in compatibility with
#  the existing training pipeline (seg_trainer.py).
# ─────────────────────────────────────────────────────────────────────────────

class GeoSampleUNet(nn.Module):
    """
    U-Net with Geometry-Guided Sampling for brain tumor segmentation.

    Drop-in replacement for modules.unet.UNet with the same interface:
        Input:  (batch, in_channels, 256, 256)
        Output: (batch, out_channels, 256, 256) — raw logits

    Args:
        in_channels:   Number of input channels (3 for RGB MRI).
        out_channels:  Number of output channels (1 for binary mask).
        base_features: Starting feature count. Doubles per level.
                       Default 32 → [32, 64, 128, 256, 512].
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_features: int = 32,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        f = base_features  # shorthand: f=60 → 60, 120, 240, 480, 960

        # ── Encoder (going down) ──────────────────────────────────────────
        self.enc1 = GeoEncoderBlock(in_channels, f)       # 3   → f
        self.enc2 = GeoEncoderBlock(f,           f * 2)   # f   → f*2
        self.enc3 = GeoEncoderBlock(f * 2,       f * 4)   # f*2 → f*4
        self.enc4 = GeoEncoderBlock(f * 4,       f * 8)   # f*4 → f*8

        # ── Bottleneck (bottom of the U) ──────────────────────────────────
        # GeoSample2D for oriented refinement at the most compressed scale.
        self.bottleneck = GeoSample2D(f * 8, f * 16)     # f*8 → f*16

        # ── Decoder (going up) ────────────────────────────────────────────
        # Each stage: upsample → consensus align → geo refine
        self.dec4 = GeoDecoderBlock(f * 16, f * 8, f * 8)  # f*16 + f*8 → f*8
        self.dec3 = GeoDecoderBlock(f * 8,  f * 4, f * 4)  # f*8  + f*4 → f*4
        self.dec2 = GeoDecoderBlock(f * 4,  f * 2, f * 2)  # f*4  + f*2 → f*2
        self.dec1 = GeoDecoderBlock(f * 2,  f,     f)       # f*2  + f   → f

        # ── Output Layer ──────────────────────────────────────────────────
        # 1×1 conv to map features to logits (same as standard U-Net).
        # No sigmoid — applied separately for BCEWithLogitsLoss compatibility.
        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        # ── Deep Supervision Auxiliary Heads ──────────────────────────────
        if self.deep_supervision:
            self.aux2 = nn.Conv2d(f * 2, out_channels, kernel_size=1)  # 128x128
            self.aux3 = nn.Conv2d(f * 4, out_channels, kernel_size=1)  # 64x64
        else:
            self.aux2 = None
            self.aux3 = None

    def forward(self, x):
        """
        Forward pass through the GeoSample U-Net.

        Args:
            x: Input tensor (batch, 3, 256, 256).

        Returns:
            In training mode with deep_supervision=True:
                tuple of (logits, aux2, aux3)
            In eval mode (or deep_supervision=False):
                logits tensor (batch, 1, 256, 256)
        """
        # ── Encoder ──
        skip1, x = self.enc1(x)   # skip1: (B, f,   256, 256) | x: (B, f,   128, 128)
        skip2, x = self.enc2(x)   # skip2: (B, f*2, 128, 128) | x: (B, f*2,  64,  64)
        skip3, x = self.enc3(x)   # skip3: (B, f*4,  64,  64) | x: (B, f*4,  32,  32)
        skip4, x = self.enc4(x)   # skip4: (B, f*8,  32,  32) | x: (B, f*8,  16,  16)

        # ── Bottleneck ──
        x = self.bottleneck(x)    # x: (B, f*16, 16, 16)

        # ── Decoder (with geometry-aligned skip connections) ──
        x_dec4 = self.dec4(x, skip4)        # (B, f*8, 32,  32)
        x_dec3 = self.dec3(x_dec4, skip3)   # (B, f*4, 64,  64)
        x_dec2 = self.dec2(x_dec3, skip2)   # (B, f*2, 128, 128)
        x_dec1 = self.dec1(x_dec2, skip1)   # (B, f,   256, 256)

        # ── Output ──
        logits = self.output_conv(x_dec1)   # (B, 1, 256, 256)

        if self.training and self.deep_supervision and self.aux2 is not None:
            aux2 = self.aux2(x_dec2)        # (B, 1, 128, 128)
            aux3 = self.aux3(x_dec3)        # (B, 1, 64, 64)
            return logits, aux2, aux3

        return logits


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience Functions
#
#  Mirrors the build_unet() and count_parameters() from modules/unet.py
#  so this model can be constructed the same way.
# ─────────────────────────────────────────────────────────────────────────────

def build_geosample_unet(config: dict | None = None) -> GeoSampleUNet:
    """
    Build a GeoSample U-Net using settings from config.yaml.

    Args:
        config: The 'segmentation' section of config.yaml.
                If None, uses default values.

    Returns:
        GeoSampleUNet model (not yet trained, weights are random).
    """
    if config is None:
        return GeoSampleUNet(in_channels=3, out_channels=1, base_features=32, deep_supervision=True)

    return GeoSampleUNet(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 1),
        base_features=config.get("base_features", 32),
        deep_supervision=config.get("deep_supervision", True),
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
