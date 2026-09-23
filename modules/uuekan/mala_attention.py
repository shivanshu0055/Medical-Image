"""
MedBoard — modules/uuekan/mala_attention.py
Magnitude-Aware Linear Attention (MALA) mechanism.

Key features:
  - Linear computational complexity: O(N) instead of quadratic O(N^2)
  - Magnitude compensation (additive beta_i + multiplicative gamma_i)
  - 2D Rotary Position Embeddings (RoPE)
  - Uncertainty mask weighting
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """Helper function for rotary position embedding (RoPE)."""
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embedding."""
    return (x * cos) + (rotate_every_two(x) * sin)


class RoPE(nn.Module):
    """2D Rotary Position Embedding for Vision feature maps."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        head_dim = embed_dim // num_heads
        # Ensure head_dim // 4 is at least 1
        num_freqs = max(1, head_dim // 4)
        angle = 1.0 / (10000 ** torch.linspace(0, 1, num_freqs))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # If angle length doesn't match head_dim // 2, pad or slice
        if angle.shape[0] < head_dim // 2:
            angle = F.pad(angle, (0, head_dim // 2 - angle.shape[0]))
        else:
            angle = angle[:head_dim // 2]
        self.register_buffer("angle", angle)

    def forward(self, slen: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = slen
        index_h = torch.arange(H, device=self.angle.device).to(self.angle.dtype)
        index_w = torch.arange(W, device=self.angle.device).to(self.angle.dtype)

        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])
        sin_h = sin_h.unsqueeze(1).repeat(1, W, 1)
        sin_w = sin_w.unsqueeze(0).repeat(H, 1, 1)
        sin = torch.cat([sin_h, sin_w], -1)

        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])
        cos_h = cos_h.unsqueeze(1).repeat(1, W, 1)
        cos_w = cos_w.unsqueeze(0).repeat(H, 1, 1)
        cos = torch.cat([cos_h, cos_w], -1)

        return sin.flatten(0, 1), cos.flatten(0, 1)


class MALAAttention(nn.Module):
    """
    Magnitude-Aware Linear Attention.
    Computes linear attention O(N) while compensating for magnitude loss.
    """

    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()
        self.rope = RoPE(dim, num_heads)

        self._cached_rope = None
        self._cached_size = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        H: int,
        W: int,
        uncertainty_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: [B, N, C] features where N = H * W
            H, W: Spatial dimensions
            uncertainty_mask: Optional [B, N, N] or [B, 1, N] uncertainty weighting
        """
        B, N, C = query.shape
        assert N == H * W, f"N={N} must equal H*W={H*W}"

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Multi-head shape: [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Non-negative mapping
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # Rotary position embeddings
        if self._cached_size != (H, W) or self._cached_rope[0].device != q.device:
            self._cached_rope = self.rope((H, W))
            self._cached_size = (H, W)
        sin, cos = self._cached_rope

        # Calculate magnitude term z
        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        # Apply RoPE
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        # Linear attention: compute (K^T * V) first => O(N * D^2)
        norm_factor = (self.scale / max(N, 1)) ** 0.5
        kv = (k.transpose(-2, -1) * norm_factor) @ (v * norm_factor)

        # MALA magnitude compensation:
        # beta_i = 1 + 1 / (z_i + eps), offset gamma_i = z_i
        res = q @ kv * (1.0 + 1.0 / (z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        if uncertainty_mask is not None:
            if uncertainty_mask.dim() == 3:
                uncertainty_mask = uncertainty_mask.unsqueeze(1)
            uncertainty_weight = uncertainty_mask.mean(dim=-1, keepdim=True)
            res = res * (1.0 + uncertainty_weight)

        res = res.transpose(1, 2).contiguous().view(B, N, C)
        return self.o_proj(res)


class MALAAttentionBlock(nn.Module):
    """Complete MALA block with LayerNorm and residual connection."""

    def __init__(self, dim: int, num_heads: int = 1, drop: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MALAAttention(dim, num_heads)
        self.drop = nn.Dropout(drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        H: int,
        W: int,
        uncertainty_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.dim() == 4:
            B, C, H, W = query.shape
            query = query.flatten(2).transpose(1, 2)
            key = key.flatten(2).transpose(1, 2)
            value = value.flatten(2).transpose(1, 2)

        q_norm = self.norm_q(query)
        k_norm = self.norm_kv(key)
        v_norm = self.norm_kv(value)

        out = self.attn(q_norm, k_norm, v_norm, H, W, uncertainty_mask)
        return self.drop(out) + query
