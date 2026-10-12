"""
MedBoard — modules/sabernet.py
Phase 2 Architecture: SaberNet (Self-Attention Based Effective Relation Network)

Reference:
  - SaberNet: Self-attention based effective relation network (Pattern Recognition, 2023).
  - BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification (arXiv benchmark).

Key Architectural Innovations:
  1. SABER Relation Module: Computes spatial self-attention relation matrices between all
     pixel coordinates across the MRI slice, modeling non-local spatial dependencies.
  2. Relation-Enhanced Skip Pathways: Enriches multi-scale skip connections with global
     contextual relations before concatenation with decoder features.
  3. Residual Relation Gating: Learnable gamma scalar initializes relation influence smoothly
     at 0, ensuring stable optimization from epoch 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Convolution Block: Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Standard double convolution block used in medical segmentation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
#  SABER Module: Self-Attention Based Effective Relation Block
# ─────────────────────────────────────────────────────────────────────────────

class SaberRelationBlock(nn.Module):
    """
    Self-Attention Based Effective Relation (SABER) Block.

    Computes a pairwise spatial relation matrix across all positions:
      Relation = Softmax( Q * K^T / sqrt(d) )
      Enhanced_Features = x + gamma * (Relation * V)

    Args:
        in_channels: Number of input channels.
        reduction:   Channel reduction factor for intermediate Q and K projections.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(in_channels // reduction, 16)

        # Q, K, V Projections
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.key_conv   = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Learnable relation scaling parameter (initialized to 0.1 for active gradient flow)
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.out_norm = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # Spatial pooling for Key and Value if spatial size is large (e.g., H > 64)
        # to ensure constant-time memory efficiency while preserving receptive field
        if H > 64:
            k_in = F.adaptive_avg_pool2d(x, (32, 32))
            K = self.key_conv(k_in).view(B, self.inter_channels, -1)   # (B, C_mid, N_k)
            V = self.value_conv(k_in).view(B, C, -1)                   # (B, C, N_k)
        else:
            K = self.key_conv(x).view(B, self.inter_channels, -1)     # (B, C_mid, N)
            V = self.value_conv(x).view(B, C, -1)                     # (B, C, N)

        Q = self.query_conv(x).view(B, self.inter_channels, -1)       # (B, C_mid, N)

        # Relation energy matrix: Q^T * K / sqrt(d)
        energy = torch.bmm(Q.transpose(1, 2), K) / (self.inter_channels ** 0.5)  # (B, N, N_k)
        attention = F.softmax(energy, dim=-1)                                    # (B, N, N_k)

        # Weight values by relation matrix
        relation_out = torch.bmm(V, attention.transpose(1, 2))                   # (B, C, N)
        relation_out = relation_out.view(B, C, H, W)

        # Residual relation bypass
        return self.out_norm(x + self.gamma * relation_out)


# ─────────────────────────────────────────────────────────────────────────────
#  SaberNet Architecture
# ─────────────────────────────────────────────────────────────────────────────

class SaberNet(nn.Module):
    """
    SaberNet: Self-Attention Based Effective Relation Network for Brain Tumor MRI.

    Combines multi-scale convolutional feature extraction with SABER self-attention
    relation modeling at the bottleneck and along skip pathways.

    Args:
        in_channels:   Input channels (3 for RGB brain MRI).
        out_channels:  Output channels (1 for binary tumor mask).
        base_features: Starting feature map channels (default 32 -> 32, 64, 128, 256, 512).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_features: int = 32,
    ):
        super().__init__()
        f = base_features

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # ── 1. Multi-Scale Encoder ───────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, f)        # Level 1: (B, 32,  256, 256)
        self.enc2 = DoubleConv(f,           f * 2)    # Level 2: (B, 64,  128, 128)
        self.enc3 = DoubleConv(f * 2,       f * 4)    # Level 3: (B, 128,  64,  64)
        self.enc4 = DoubleConv(f * 4,       f * 8)    # Level 4: (B, 256,  32,  32)

        # ── 2. SABER Relation Bottleneck (16×16) ─────────────────────────────
        self.bottleneck_conv  = DoubleConv(f * 8, f * 16)               # (B, 512, 16, 16)
        self.saber_bottleneck = SaberRelationBlock(f * 16, reduction=4) # Global self-attention

        # ── 3. SABER Skip Connection Enhancers ───────────────────────────────
        self.saber_skip4 = SaberRelationBlock(f * 8, reduction=4)  # 256 channels
        self.saber_skip3 = SaberRelationBlock(f * 4, reduction=4)  # 128 channels
        self.saber_skip2 = SaberRelationBlock(f * 2, reduction=4)  # 64 channels
        self.saber_skip1 = SaberRelationBlock(f,     reduction=2)  # 32 channels

        # ── 4. Decoder Stages ────────────────────────────────────────────────
        self.dec4 = DoubleConv(f * 16 + f * 8, f * 8)  # 512 + 256 -> 256
        self.dec3 = DoubleConv(f * 8  + f * 4, f * 4)  # 256 + 128 -> 128
        self.dec2 = DoubleConv(f * 4  + f * 2, f * 2)  # 128 + 64  -> 64
        self.dec1 = DoubleConv(f * 2  + f,     f)      # 64  + 32  -> 32

        # ── 5. Output Head ───────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SaberNet.

        Args:
            x: Input tensor of shape (B, 3, 256, 256).

        Returns:
            Logits tensor of shape (B, 1, 256, 256).
        """
        # ── Encoder ──
        s1 = self.enc1(x)               # (B, 32,  256, 256)
        s2 = self.enc2(self.pool(s1))   # (B, 64,  128, 128)
        s3 = self.enc3(self.pool(s2))   # (B, 128,  64,  64)
        s4 = self.enc4(self.pool(s3))   # (B, 256,  32,  32)

        # ── Bottleneck with SABER Global Self-Attention ──
        b = self.bottleneck_conv(self.pool(s4))  # (B, 512, 16, 16)
        b = self.saber_bottleneck(b)             # Non-local relation modeling

        # ── Relation-Enhanced Skips ──
        s4_rel = self.saber_skip4(s4)
        s3_rel = self.saber_skip3(s3)
        s2_rel = self.saber_skip2(s2)
        s1_rel = self.saber_skip1(s1)

        # ── Decoder ──
        d4 = self.dec4(torch.cat([self.up(b),  s4_rel], dim=1))  # (B, 256, 32,  32)
        d3 = self.dec3(torch.cat([self.up(d4), s3_rel], dim=1))  # (B, 128, 64,  64)
        d2 = self.dec2(torch.cat([self.up(d3), s2_rel], dim=1))  # (B, 64,  128, 128)
        d1 = self.dec1(torch.cat([self.up(d2), s1_rel], dim=1))  # (B, 32,  256, 256)

        # ── Output ──
        return self.output_conv(d1)  # (B, 1, 256, 256)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers & Factory Function
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_sabernet(config: dict | None = None) -> SaberNet:
    """
    Factory function to build SaberNet from config dictionary.
    """
    if config is None:
        config = {}

    return SaberNet(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 1),
        base_features=config.get("base_features", 32),
    )
