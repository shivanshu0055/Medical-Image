"""
MedBoard — modules/unet_plusplus_resnet.py
Phase 2 Architecture: Pretrained U-Net++ (ResNet-34 Backbone)

Architecture:
  - Encoder: ImageNet-pretrained ResNet-34 backbone with residual blocks.
  - Receptive Field Preservation: Dilated conv in layer4 preserves 16×16 spatial
    resolution at the bottleneck without sub-sampling artifacts.
  - Nested Decoder: 5-level dense nested skip pathway grid (Zhou et al., 2018).
  - Deep Supervision: Multi-scale auxiliary loss heads at X_0,1, X_0,2, X_0,3, X_0,4.
  - Two-Speed Optimization: Supports differential learning rate groups
    (encoder_lr for gentle adaptation, decoder_lr for rapid mask learning).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights


# ─────────────────────────────────────────────────────────────────────────────
#  Convolution Block: Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Standard double convolution block used in U-Net++ dense pathways.
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
#  Pretrained U-Net++ with ResNet-34 Backbone
# ─────────────────────────────────────────────────────────────────────────────

class ResNet34UNetPlusPlus(nn.Module):
    """
    U-Net++ with ImageNet-pretrained ResNet-34 encoder backbone.

    Encoder Stages (from ResNet-34):
      Level 0 (256x256): Initial DoubleConv stem (3 -> 32 channels)
      Level 1 (128x128): ResNet Stem (conv1 + bn1 + relu) -> 64 channels
      Level 2 (64x64):   ResNet Layer1 (after maxpool)    -> 64 channels
      Level 3 (32x32):   ResNet Layer2                    -> 128 channels
      Level 4 (16x16):   ResNet Layer3 + Dilated Layer4   -> 512 channels (Bottleneck)

    Dense Nested Decoder:
      Standard U-Net++ dense grid connecting all skip features to decoder nodes.
      Deep supervision across top heads (X_0,1 to X_0,4).

    Args:
        in_channels:      Input image channels (3 for RGB brain MRI).
        out_channels:     Output segmentation channels (1 for binary tumor mask).
        pretrained:       Whether to load official ImageNet weights for ResNet-34.
        deep_supervision: If True, outputs list of 4 heads during training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # ── 1. Pretrained ResNet-34 Encoder Backbone ─────────────────────────
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet34(weights=weights)

        # Receptive Field Preservation:
        # Dilate layer4 so it outputs at 16x16 instead of collapsing to 8x8.
        # This keeps high spatial resolution at the bottleneck.
        for m in self.resnet.layer4.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (2, 2)
                    m.padding = (2, 2)

        # Level 0 Stem (full resolution 256×256)
        self.enc0 = DoubleConv(in_channels, 32)

        # Channel progression across levels:
        # nb[0] = 32  (Level 0: 256x256)
        # nb[1] = 64  (Level 1: 128x128)
        # nb[2] = 64  (Level 2: 64x64)
        # nb[3] = 128 (Level 3: 32x32)
        # nb[4] = 512 (Level 4: 16x16 - Bottleneck)
        nb = [32, 64, 64, 128, 512]

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # ── 2. Column j = 1 ──────────────────────────────────────────────────
        self.conv0_1 = DoubleConv(nb[0] + nb[1], nb[0])
        self.conv1_1 = DoubleConv(nb[1] + nb[2], nb[1])
        self.conv2_1 = DoubleConv(nb[2] + nb[3], nb[2])
        self.conv3_1 = DoubleConv(nb[3] + nb[4], nb[3])

        # ── 3. Column j = 2 ──────────────────────────────────────────────────
        self.conv0_2 = DoubleConv(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = DoubleConv(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = DoubleConv(nb[2] * 2 + nb[3], nb[2])

        # ── 4. Column j = 3 ──────────────────────────────────────────────────
        self.conv0_3 = DoubleConv(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = DoubleConv(nb[1] * 3 + nb[2], nb[1])

        # ── 5. Column j = 4 ──────────────────────────────────────────────────
        self.conv0_4 = DoubleConv(nb[0] * 4 + nb[1], nb[0])

        # ── 6. Output Heads (Deep Supervision) ───────────────────────────────
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """
        Forward pass through pretrained ResNet-34 encoder and dense nested decoder.
        """
        # ── Encoder (Column j = 0) ───────────────────────────────────────────
        x0_0 = self.enc0(x)                                         # (B, 32,  256, 256)
        x1_0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))  # (B, 64,  128, 128)
        x2_0 = self.resnet.layer1(self.resnet.maxpool(x1_0))        # (B, 64,   64,  64)
        x3_0 = self.resnet.layer2(x2_0)                             # (B, 128,  32,  32)
        x4_0 = self.resnet.layer4(self.resnet.layer3(x3_0))         # (B, 512,  16,  16)

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

        # ── Output Heads ─────────────────────────────────────────────────────
        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            if self.training:
                return [out1, out2, out3, out4]
            return out4

        return self.final(x0_4)

    def get_parameter_groups(
        self,
        encoder_lr: float = 6e-5,
        decoder_lr: float = 1.8e-4,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """
        Create differential learning rate parameter groups for Adam optimizer.
        - Encoder: Trains at encoder_lr to protect pretrained ImageNet representations.
        - Decoder: Trains at decoder_lr for faster convergence of new skip pathways.
        """
        encoder_params = list(self.resnet.parameters())
        encoder_param_ids = set(map(id, encoder_params))

        decoder_params = [
            p for p in self.parameters()
            if id(p) not in encoder_param_ids and p.requires_grad
        ]

        return [
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay},
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  Helper & Factory Function
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_resnet34_unet_plusplus(config: dict | None = None) -> ResNet34UNetPlusPlus:
    """
    Factory function to build ResNet34UNetPlusPlus.
    """
    if config is None:
        config = {}

    return ResNet34UNetPlusPlus(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 1),
        pretrained=config.get("pretrained", True),
        deep_supervision=config.get("deep_supervision", True),
    )
