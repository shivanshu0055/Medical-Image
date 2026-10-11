"""
MedBoard — modules/unet_plusplus.py
Phase 2 Architecture: U-Net++ (Nested U-Net)

Reference:
  Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018).
  "UNet++: A Nested U-Net Architecture for Medical Image Segmentation."
  Deep Learning in Medical Image Analysis (DLMIA), Springer LNCS.
  IEEE Transactions on Medical Imaging (TMI), 2019.

Key architectural concepts:
  1. Dense Skip Pathways: Connects encoder and decoder through an intermediate
     triangular grid of nested convolution blocks to narrow the semantic gap.
  2. Multi-Scale Feature Aggregation: Captures tumor features at various receptive
     fields simultaneously.
  3. Deep Supervision: Computes loss over multiple resolution heads (X_0,1, X_0,2,
     X_0,3, X_0,4) during training to enforce direct gradient flow into earlier layers.
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
    Standard double convolution block used in U-Net and U-Net++.
    Consists of two consecutive 3×3 convolutions, each followed by
    Batch Normalization and ReLU activation.
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
#  U-Net++ (Nested U-Net) Architecture
# ─────────────────────────────────────────────────────────────────────────────

class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) for brain tumor segmentation.

    Nodes:
      Level 0 (256x256):   x0_0 ──→ x0_1 ──→ x0_2 ──→ x0_3 ──→ x0_4
      Level 1 (128x128):   x1_0 ──→ x1_1 ──→ x1_2 ──→ x1_3
      Level 2 (64x64):     x2_0 ──→ x2_1 ──→ x2_2
      Level 3 (32x32):     x3_0 ──→ x3_1
      Level 4 (16x16):     x4_0 (Bottleneck)

    Args:
        in_channels:      Input image channels (3 for RGB MRI).
        out_channels:     Output segmentation channels (1 for binary tumor mask).
        base_features:    Starting filter count at Level 0 (default 32 → [32, 64, 128, 256, 512]).
        deep_supervision: If True, outputs a list of 4 predictions during training [x0_1, x0_2, x0_3, x0_4].
                          If False or during eval, outputs only the final refined prediction (x0_4).
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
        self.base_features = base_features

        # Filter progression: e.g. [32, 64, 128, 256, 512]
        nb = [base_features * (2 ** i) for i in range(5)]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # ── Column j = 0 (Encoder Backbone) ──────────────────────────────────
        self.conv0_0 = DoubleConv(in_channels, nb[0])
        self.conv1_0 = DoubleConv(nb[0],       nb[1])
        self.conv2_0 = DoubleConv(nb[1],       nb[2])
        self.conv3_0 = DoubleConv(nb[2],       nb[3])
        self.conv4_0 = DoubleConv(nb[3],       nb[4])

        # ── Column j = 1 ─────────────────────────────────────────────────────
        self.conv0_1 = DoubleConv(nb[0] + nb[1], nb[0])
        self.conv1_1 = DoubleConv(nb[1] + nb[2], nb[1])
        self.conv2_1 = DoubleConv(nb[2] + nb[3], nb[2])
        self.conv3_1 = DoubleConv(nb[3] + nb[4], nb[3])

        # ── Column j = 2 ─────────────────────────────────────────────────────
        self.conv0_2 = DoubleConv(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = DoubleConv(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = DoubleConv(nb[2] * 2 + nb[3], nb[2])

        # ── Column j = 3 ─────────────────────────────────────────────────────
        self.conv0_3 = DoubleConv(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = DoubleConv(nb[1] * 3 + nb[2], nb[1])

        # ── Column j = 4 ─────────────────────────────────────────────────────
        self.conv0_4 = DoubleConv(nb[0] * 4 + nb[1], nb[0])

        # ── Output 1×1 Convolutions (Deep Supervision Heads) ─────────────────
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """
        Forward pass through nested grid.

        Args:
            x: Input tensor (B, 3, 256, 256).

        Returns:
            If deep_supervision=True and training:
                [out1, out2, out3, out4], each of shape (B, 1, 256, 256).
            Otherwise:
                Logits tensor of shape (B, 1, 256, 256) from the final node (x0_4).
        """
        # ── Column j = 0 (Encoder) ───────────────────────────────────────────
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # ── Column j = 1 ─────────────────────────────────────────────────────
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # ── Column j = 2 ─────────────────────────────────────────────────────
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # ── Column j = 3 ─────────────────────────────────────────────────────
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # ── Column j = 4 ─────────────────────────────────────────────────────
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        # ── Outputs ──────────────────────────────────────────────────────────
        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            if self.training:
                return [out1, out2, out3, out4]
            return out4

        return self.final(x0_4)


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience Helpers & Factory Function
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_unet_plusplus(config: dict | None = None) -> UNetPlusPlus:
    """
    Factory function to build UNetPlusPlus from config dictionary.

    Args:
        config: Dictionary containing model hyperparams, e.g.:
                {
                    'in_channels': 3,
                    'out_channels': 1,
                    'base_features': 32,
                    'deep_supervision': True
                }

    Returns:
        Configured UNetPlusPlus model instance.
    """
    if config is None:
        config = {}

    return UNetPlusPlus(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 1),
        base_features=config.get("base_features", 32),
        deep_supervision=config.get("deep_supervision", True),
    )
