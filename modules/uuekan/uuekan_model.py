"""
MedBoard — modules/uuekan/uuekan_model.py
Complete UUEKAN Architecture:
Edge-Enhanced Kolmogorov-Arnold Network with Uncertainty-Guided Linear Attention.

Combines:
  - Shallow convolutional stages (1-3)
  - Deep EKAN stages (4-5) with learnable B-splines and Sobel edge injection
  - U-MALA skip connections with Uncertainty Map Generation
  - Auxiliary heads for multi-stage Deep Supervision
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Union

from .kan import KANLinear
from .uncertainty import (
    UncertaintyGuidedProcessor,
    UncertaintyRefinementAttention,
)


class DW_bn_relu(nn.Module):
    """Depthwise Convolution + BatchNorm + ReLU for spatial mixing."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x.flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):
    """Image / Feature map to Patch Embedding."""

    def __init__(self, in_chans: int = 128, embed_dim: int = 160, patch_size: int = 3, stride: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ConvLayer(nn.Module):
    """Encoder double convolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class D_ConvLayer(nn.Module):
    """Decoder convolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FeatureEdgeExtractor(nn.Module):
    """
    Extracts semantic boundaries using fixed Sobel kernels + learnable fusion & attention.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.sobel_x = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.sobel_y = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        for i in range(dim):
            self.sobel_x.weight.data[i, 0] = sobel_kernel_x
            self.sobel_y.weight.data[i, 0] = sobel_kernel_y

        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

        self.edge_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        mid_dim = max(dim // 4, 16)
        self.edge_attention = nn.Sequential(
            nn.Conv2d(dim, mid_dim, kernel_size=1),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x_spatial = x.transpose(1, 2).view(B, C, H, W)

        edge_x = self.sobel_x(x_spatial)
        edge_y = self.sobel_y(x_spatial)

        edge_combined = torch.cat([edge_x, edge_y], dim=1)
        edge_features = self.edge_fusion(edge_combined)
        attention = self.edge_attention(edge_features)
        edge_features = edge_features * attention

        return edge_features.flatten(2).transpose(1, 2)


class KANLayer(nn.Module):
    """
    Alternating KANLinear layers and Depthwise Separable Convolutions.
    """

    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        self.fc1 = KANLinear(in_features, hidden_features, grid_size=5, spline_order=3)
        self.dwconv_1 = DW_bn_relu(hidden_features)

        self.fc2 = KANLinear(hidden_features, out_features, grid_size=5, spline_order=3)
        self.dwconv_2 = DW_bn_relu(out_features)

        self.fc3 = KANLinear(out_features, out_features, grid_size=5, spline_order=3)
        self.dwconv_3 = DW_bn_relu(out_features)

        self.fc4 = KANLinear(out_features, out_features, grid_size=5, spline_order=3)
        self.dwconv_4 = DW_bn_relu(out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, -1).contiguous()
        x = self.dwconv_1(x, H, W)

        x = self.fc2(x.reshape(B * N, -1)).reshape(B, N, -1).contiguous()
        x = self.dwconv_2(x, H, W)

        x = self.fc3(x.reshape(B * N, -1)).reshape(B, N, -1).contiguous()
        x = self.dwconv_3(x, H, W)

        x = self.fc4(x.reshape(B * N, -1)).reshape(B, N, -1).contiguous()
        x = self.dwconv_4(x, H, W)
        return self.drop(x)


class EKANBlock(nn.Module):
    """Edge-Enhanced KAN Block combining Edge Extractor with KANLayer."""

    def __init__(self, dim: int, drop: float = 0.0, edge_weight: float = 0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.edge_weight = edge_weight

        self.edge_extractor = FeatureEdgeExtractor(dim)
        self.layer = KANLayer(in_features=dim, hidden_features=dim, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        edge_feat = self.edge_extractor(self.norm1(x), H, W)
        fused = x + self.edge_weight * edge_feat
        out = fused + self.layer(self.norm2(fused), H, W)
        return out


class UUEKAN(nn.Module):
    """
    UUEKAN Network for Medical Image Segmentation.

    Args:
        in_channels: Number of image channels (3 for RGB MRI slices).
        num_classes: Number of target mask classes (1 for binary tumor mask).
        img_size: Input spatial size (default: 512).
        embed_dims: Feature dimensions for stages [Stage 3, Stage 4, Stage 5]
                    Default: [128, 160, 256] matching paper.
        drop_rate: Dropout rate.
        use_uncertainty: Whether to apply U-MALA uncertainty-guided skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 512,
        embed_dims: list[int] = [128, 160, 256],
        drop_rate: float = 0.1,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty

        c0 = embed_dims[0] // 4   # 32
        c1 = embed_dims[0] // 2   # 64
        c2 = embed_dims[0]        # 128
        c3 = embed_dims[1]        # 160
        c4 = embed_dims[2]        # 256

        # ─── Shallow Conv Encoder (Stages 1 to 3) ───────────────────────────
        self.encoder1 = ConvLayer(in_channels, c0)  # (B, 32, 512, 512)
        self.encoder2 = ConvLayer(c0, c1)          # (B, 64, 256, 256)
        self.encoder3 = ConvLayer(c1, c2)          # (B, 128, 128, 128)

        # ─── Deep EKAN Encoder (Stages 4 & 5) ───────────────────────────────
        self.patch_embed3 = PatchEmbed(in_chans=c2, embed_dim=c3, patch_size=3, stride=2)
        self.block1 = nn.ModuleList([EKANBlock(dim=c3, drop=drop_rate)])
        self.norm3 = nn.LayerNorm(c3)

        self.patch_embed4 = PatchEmbed(in_chans=c3, embed_dim=c4, patch_size=3, stride=2)
        self.block2 = nn.ModuleList([EKANBlock(dim=c4, drop=drop_rate)])
        self.norm4 = nn.LayerNorm(c4)

        # ─── Decoder Stages ─────────────────────────────────────────────────
        self.decoder1 = D_ConvLayer(c4, c3)
        self.dblock1  = nn.ModuleList([EKANBlock(dim=c3, drop=drop_rate)])
        self.dnorm1   = nn.LayerNorm(c3)

        self.decoder2 = D_ConvLayer(c3, c2)
        self.dblock2  = nn.ModuleList([EKANBlock(dim=c2, drop=drop_rate)])
        self.dnorm2   = nn.LayerNorm(c2)

        self.decoder3 = D_ConvLayer(c2, c1)
        self.decoder4 = D_ConvLayer(c1, c0)
        self.decoder5 = D_ConvLayer(c0, c0)

        # Final prediction head
        self.final = nn.Conv2d(c0, num_classes, kernel_size=1)

        # ─── Auxiliary Prediction Heads for Deep Supervision ─────────────────
        self.aux_head4 = nn.Conv2d(c3, num_classes, kernel_size=1)
        self.aux_head3 = nn.Conv2d(c2, num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.aux_head1 = nn.Conv2d(c0, num_classes, kernel_size=1)

        # ─── Uncertainty Processor & Skip Attention Modules ──────────────────
        if self.use_uncertainty:
            self.uncertainty_processor = UncertaintyGuidedProcessor(base_size=(img_size, img_size))

            self.u_conv4 = nn.Conv2d(c3, 1, kernel_size=1)
            self.u_conv3 = nn.Conv2d(c2, 1, kernel_size=1)
            self.u_conv2 = nn.Conv2d(c1, 1, kernel_size=1)
            self.u_conv1 = nn.Conv2d(c0, 1, kernel_size=1)

            self.u_attn4 = UncertaintyRefinementAttention(in_channel=c3, dim=c3, stage=4)
            self.u_attn3 = UncertaintyRefinementAttention(in_channel=c2, dim=c2, stage=3)
            self.u_attn2 = UncertaintyRefinementAttention(in_channel=c1, dim=c1, stage=2)
            self.u_attn1 = UncertaintyRefinementAttention(in_channel=c0, dim=c0, stage=1)

    def forward(
        self,
        x: torch.Tensor,
        auxiliary: bool = False,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """
        Args:
            x: Input tensor [B, 3, H, W]
            auxiliary: If True, return main mask + aux masks + uncertainty maps.
            return_dict: If True, return results packaged in a dictionary.
        """
        B, _, H_orig, W_orig = x.shape

        # ─── Encoder ────────────────────────────────────────────────────────
        t1 = self.encoder1(x)                                    # (B, 32, 512, 512)
        out1 = F.relu(F.max_pool2d(t1, 2, 2))                    # (B, 32, 256, 256)

        t2 = self.encoder2(out1)                                  # (B, 64, 256, 256)
        out2 = F.relu(F.max_pool2d(t2, 2, 2))                    # (B, 64, 128, 128)

        t3 = self.encoder3(out2)                                  # (B, 128, 128, 128)
        out3 = F.relu(F.max_pool2d(t3, 2, 2))                    # (B, 128, 64, 64)

        # Stage 4: EKAN
        out4, H4, W4 = self.patch_embed3(out3)                   # (B, 1024, 160)
        for blk in self.block1:
            out4 = blk(out4, H4, W4)
        out4 = self.norm3(out4)
        t4 = out4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous() # (B, 160, 32, 32)

        # Stage 5: EKAN Bottleneck
        out5, H5, W5 = self.patch_embed4(t4)
        for blk in self.block2:
            out5 = blk(out5, H5, W5)
        out5 = self.norm4(out5)
        bottleneck = out5.reshape(B, H5, W5, -1).permute(0, 3, 1, 2).contiguous() # (B, 256, 16, 16)

        # ─── Decoder with U-MALA Skip Connections ───────────────────────────
        aux_outputs: List[torch.Tensor] = []
        uncertainty_maps: List[torch.Tensor] = []

        # Decoder Stage 4 (16x16 -> 32x32)
        d4 = F.relu(F.interpolate(self.decoder1(bottleneck), scale_factor=(2, 2), mode="bilinear", align_corners=False))
        if self.use_uncertainty:
            u_sal4 = self.u_conv4(d4)
            u_map4 = self.uncertainty_processor.process(u_sal4, (d4.shape[2], d4.shape[3]))
            refined_t4, _, _ = self.u_attn4(t4, d4, u_map4)
            d4 = d4 + refined_t4
            uncertainty_maps.append(u_map4)
        else:
            d4 = d4 + t4

        d4_tokens = d4.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            d4_tokens = blk(d4_tokens, H4, W4)
        d4 = self.dnorm1(d4_tokens).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        if auxiliary:
            aux_outputs.append(F.interpolate(self.aux_head4(d4), size=(H_orig, W_orig), mode="bilinear", align_corners=False))

        # Decoder Stage 3 (32x32 -> 64x64)
        d3 = F.relu(F.interpolate(self.decoder2(d4), scale_factor=(2, 2), mode="bilinear", align_corners=False))
        if self.use_uncertainty:
            u_sal3 = self.u_conv3(d3)
            u_map3 = self.uncertainty_processor.process(u_sal3, (d3.shape[2], d3.shape[3]))
            refined_t3, _, _ = self.u_attn3(out3, d3, u_map3)
            d3 = d3 + refined_t3
            uncertainty_maps.append(u_map3)
        else:
            d3 = d3 + out3

        d3_tokens = d3.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            d3_tokens = blk(d3_tokens, out3.shape[2], out3.shape[3])
        d3 = self.dnorm2(d3_tokens).reshape(B, out3.shape[2], out3.shape[3], -1).permute(0, 3, 1, 2).contiguous()
        if auxiliary:
            aux_outputs.append(F.interpolate(self.aux_head3(d3), size=(H_orig, W_orig), mode="bilinear", align_corners=False))

        # Decoder Stage 2 (64x64 -> 128x128)
        d2 = F.relu(F.interpolate(self.decoder3(d3), scale_factor=(2, 2), mode="bilinear", align_corners=False))
        if self.use_uncertainty:
            u_sal2 = self.u_conv2(d2)
            u_map2 = self.uncertainty_processor.process(u_sal2, (d2.shape[2], d2.shape[3]))
            refined_t2, _, _ = self.u_attn2(out2, d2, u_map2)
            d2 = d2 + refined_t2
            uncertainty_maps.append(u_map2)
        else:
            d2 = d2 + out2
        if auxiliary:
            aux_outputs.append(F.interpolate(self.aux_head2(d2), size=(H_orig, W_orig), mode="bilinear", align_corners=False))

        # Decoder Stage 1 (128x128 -> 256x256)
        d1 = F.relu(F.interpolate(self.decoder4(d2), scale_factor=(2, 2), mode="bilinear", align_corners=False))
        if self.use_uncertainty:
            u_sal1 = self.u_conv1(d1)
            u_map1 = self.uncertainty_processor.process(u_sal1, (d1.shape[2], d1.shape[3]))
            refined_t1, _, _ = self.u_attn1(out1, d1, u_map1)
            d1 = d1 + refined_t1
            uncertainty_maps.append(u_map1)
        else:
            d1 = d1 + out1
        if auxiliary:
            aux_outputs.append(F.interpolate(self.aux_head1(d1), size=(H_orig, W_orig), mode="bilinear", align_corners=False))

        # Final reconstruction (256x256 -> 512x512)
        d0 = F.relu(F.interpolate(self.decoder5(d1), scale_factor=(2, 2), mode="bilinear", align_corners=False))
        final_output = self.final(d0)  # (B, 1, 512, 512)

        if return_dict:
            return {
                "pred": final_output,
                "aux_preds": aux_outputs,
                "uncertainty_maps": uncertainty_maps,
            }

        if auxiliary:
            return final_output, aux_outputs, uncertainty_maps

        return final_output


def build_uuekan(config: dict | None = None) -> UUEKAN:
    """Helper factory function to instantiate UUEKAN from config dictionary."""
    config = config or {}
    model_cfg = config.get("model", config)
    return UUEKAN(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 1),
        img_size=model_cfg.get("img_size", 512),
        embed_dims=model_cfg.get("embed_dims", [128, 160, 256]),
        drop_rate=model_cfg.get("drop_rate", 0.1),
        use_uncertainty=model_cfg.get("use_uncertainty", True),
    )
