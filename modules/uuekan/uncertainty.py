"""
MedBoard — modules/uuekan/uncertainty.py
Uncertainty Map Generation (UMG) and Uncertainty-Aware Dynamic Partitioning (UDP).

Features:
  - Pure PyTorch Gaussian smoothing kernel (no OpenCV CPU conversion needed)
  - Distance-to-decision-boundary metric: U = tau - |P - tau| (tau=0.5)
  - Dynamic sub-graph Quadtree partitioning
  - MALA-powered uncertainty refinement attention
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .mala_attention import MALAAttention


def get_gaussian_kernel_2d(ksize: int = 7, sigma: float = 1.0) -> torch.Tensor:
    """Creates a normalized 2D Gaussian filter kernel in PyTorch."""
    coords = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    gauss = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, ksize, ksize)


class UncertaintyMapGenerator(nn.Module):
    """
    Quantifies pixel-wise ambiguity:
      P = Sigmoid(Conv1x1(F_D))
      U = tau - |P - tau|   (tau=0.5)
      U_smoothed = Gaussian_filter(U)
      U_normalized = (clip(U, alpha, beta) - alpha) / (beta - alpha)
    """

    def __init__(
        self,
        ksize: int = 7,
        sigma: float = 1.0,
        channels: int = 1,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels
        self.uthreshold = threshold

        kernel = get_gaussian_kernel_2d(ksize, sigma).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(
        self, saliency_map: torch.Tensor, target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
            saliency_map: [B, 1, H_in, W_in] or logits
            target_shape: (H, W) to match
        Returns:
            uncertainty_map: [B, 1, H, W] in range [0, 1]
        """
        # Interpolate and ensure in probability space [0, 1]
        smap = F.interpolate(
            saliency_map, size=target_shape, mode="bilinear", align_corners=False
        )
        if smap.min() < 0.0 or smap.max() > 1.0:
            smap = torch.sigmoid(smap)

        # Distance from decision boundary
        p = smap - self.uthreshold
        uncertainty = self.uthreshold - torch.abs(p)  # peak uncertainty at 0.5 is 0.5

        # Gaussian smoothing
        pad = self.ksize // 2
        padded_u = F.pad(uncertainty, (pad, pad, pad, pad), mode="reflect")
        kernel = self.kernel.to(device=padded_u.device, dtype=padded_u.dtype)
        smoothed = F.conv2d(padded_u, kernel, groups=self.channels)

        # Percentile/min-max normalization to [0, 1]
        b = smoothed.shape[0]
        smoothed_flat = smoothed.view(b, -1)
        s_min = smoothed_flat.min(dim=-1, keepdim=True)[0].view(b, 1, 1, 1)
        s_max = smoothed_flat.max(dim=-1, keepdim=True)[0].view(b, 1, 1, 1)
        denom = (s_max - s_min).clamp(min=1e-6)
        normalized = (smoothed - s_min) / denom

        return normalized.clamp(0.0, 1.0)


class AdaptivePartition(nn.Module):
    """
    Uncertainty-Aware Dynamic Sub-graph Partitioning (UDP).
    Evaluates whether a region should be divided into quadrants.
    """

    def __init__(self, base_size: Tuple[int, int], pthreshold: float = 0.4, min_size: int = 8):
        super().__init__()
        self.base_size = base_size
        self.pthreshold = pthreshold
        self.min_size = min_size

    def should_partition(self, u_window: torch.Tensor, current_size: int) -> bool:
        if current_size <= self.min_size:
            return False
        avg_uncertainty = u_window.mean().item()
        # If uncertainty is low and window size allows, partition into finer details
        return avg_uncertainty < self.pthreshold


class UncertaintyRefinementAttention(nn.Module):
    """
    U-MALA Skip Connection Refinement Module.
    Combines encoder shallow features (Query) with decoder deep features (Key, Value)
    guided by the spatial uncertainty map.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int = 1,
        dim: int = 128,
        base_size: Tuple[int, int] = (64, 64),
        stage: int = 1,
        use_mala: bool = True,
    ):
        super().__init__()
        self.base_size = base_size
        self.dim = dim
        self.stage = stage
        self.use_mala = use_mala

        self.norm = nn.BatchNorm2d(dim)
        self.lnorm = nn.BatchNorm2d(dim)

        if use_mala:
            self.attention = MALAAttention(dim, num_heads=1)
        else:
            self.mha = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

        self.conv_out1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv_out3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.conv_out4 = nn.Conv2d(dim, out_channel, kernel_size=1)

    def forward(
        self,
        f_enc: torch.Tensor,
        f_dec: torch.Tensor,
        uncertainty_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f_enc: [B, C, H, W] Shallow encoder features
            f_dec: [B, C, H, W] Deep decoder features
            uncertainty_map: [B, 1, H, W]
        Returns:
            refined_features: [B, C, H, W]
            side_output: [B, 1, H, W] (for auxiliary deep supervision)
            uncertainty_map: [B, 1, H, W]
        """
        B, C, H, W = f_enc.shape
        N = H * W

        f_enc_norm = self.norm(f_enc)
        f_dec_norm = self.lnorm(f_dec)

        q = f_enc_norm.flatten(2).transpose(1, 2)  # [B, N, C]
        k = f_dec_norm.flatten(2).transpose(1, 2)
        v = f_dec_norm.flatten(2).transpose(1, 2)
        u = uncertainty_map.flatten(2).transpose(1, 2)  # [B, N, 1]

        if self.use_mala:
            refined = self.attention(q, k, v, H, W, uncertainty_mask=u)
        else:
            refined, _ = self.mha(q, k, v)

        refined = refined.transpose(1, 2).contiguous().view(B, C, H, W)
        refined = self.conv_out1(refined)
        refined = self.conv_out3(refined)

        # Auxiliary side prediction head for deep supervision
        side_output = self.conv_out4(refined)

        return refined, side_output, uncertainty_map


class UncertaintyGuidedProcessor(nn.Module):
    """Processor that generates and manages multi-scale uncertainty maps."""

    def __init__(self, base_size: Tuple[int, int] = (512, 512), uncertainty_threshold: float = 0.5):
        super().__init__()
        self.base_size = base_size
        self.uncertainty_generator = UncertaintyMapGenerator(
            ksize=7, sigma=1.0, channels=1, threshold=uncertainty_threshold
        )

    def process(self, saliency_map: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        return self.uncertainty_generator(saliency_map, target_shape)
