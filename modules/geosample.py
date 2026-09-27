"""
MedBoard — modules/geosample.py
GeoSample: Geometry-Guided Sampling Operators for 2D Segmentation

Adapted from "Steer the Sampling, Not the Kernel Grid: Geometry-Guided
Sampling Operator for Volumetric Segmentation" (2025) for 2D brain MRI slices.

Instead of rigid 3×3 convolution grids, GeoSample:
  1. Predicts a local 2D orientation (angle) and step sizes at every pixel.
  2. Projects features to a compact bottleneck dimension for efficient sampling.
  3. Samples 4 symmetric neighbor probes aligned to the learned boundary direction.
  4. Extracts explicit differential tokens (context, gradient, curvature).
  5. Mixes tokens with lightweight 1×1 convolutions back to output channels.

This replaces heavy 3×3 convolutions with boundary-aware, geometry-steered
sampling that captures edges, gradients, and curvatures explicitly — using
roughly half the parameters of a standard convolution block.

Key modules:
  - GeometryHead2D:    Predicts per-pixel orientation + step sizes.
  - GeoSample2D:       Stride-1 refinement block (replaces DoubleConv).
  - GeoDownsample2D:   Stride-2 downsampler (replaces MaxPool2d).
  - ConsensusSkip2D:   Skip-connection alignment (replaces naive concatenation).

Design note — Bottleneck Projection:
  The geometry head operates on FULL input channels (it only outputs 4 values,
  so it's always cheap: C_in * 4 parameters). But the symmetric sampling and
  differential token extraction happen on a PROJECTED bottleneck dimension
  (C_in // reduction, clamped to [min_inner, max_inner]). This prevents the
  6x token expansion from creating an explosion at high channel counts while
  still letting the geometry prediction see all available feature information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
#
#  r_min / r_max: The minimum and maximum step sizes (in pixels) for the
#  oriented probes. These bound how far from each pixel the network is
#  allowed to look along its learned directions.
#    - r_min = 0.5: Ensures the probe always reaches at least half a pixel
#      away (avoids degenerate zero-offset sampling).
#    - r_max = 3.0: Limits reach to a local 3-pixel neighborhood, preventing
#      the probe from skipping over fine anatomical structures.
#
#  eps: Small constant to prevent division by zero in gradient / curvature
#  computations.
#
#  Bottleneck defaults:
#    - REDUCTION = 4:    Inner dim = C_in // 4.
#    - MIN_INNER = 8:    Floor for very small input channels (e.g. 3-channel RGB).
#    - MAX_INNER = 64:   Ceiling to prevent decoder blocks (768-ch concat) from
#                        inflating the inner dim. Keeps token channels <= 384.
# ─────────────────────────────────────────────────────────────────────────────

R_MIN     = 0.5
R_MAX     = 3.0
EPS       = 1e-6
REDUCTION = 4
MIN_INNER = 8
MAX_INNER = 64


def _compute_inner_dim(channels: int) -> int:
    """Compute bottleneck inner dimension with floor and ceiling clamps."""
    return max(min(channels // REDUCTION, MAX_INNER), MIN_INNER)


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry Head
#
#  A tiny 1×1 convolution that predicts, at every pixel:
#    - 2 values for direction: raw (dx, dy) → normalized to unit vector
#      [cos(theta), sin(theta)]. The orthogonal direction is computed
#      analytically as [-sin(theta), cos(theta)].
#    - 2 values for step sizes: raw scalars → sigmoid → scaled to [r_min, r_max].
#
#  Total output: 4 channels per pixel, extremely lightweight.
#  Parameter cost: C_in * 4 + 4 (bias) — negligible at any channel count.
#
#  Why 1×1 conv? The geometry should be predicted from the feature values at
#  each pixel (channel information), not from spatial neighbors. Spatial
#  context is already captured by the features flowing through the network.
# ─────────────────────────────────────────────────────────────────────────────

class GeometryHead2D(nn.Module):
    """
    Predicts per-pixel 2D orientation and step sizes from input features.

    Input:  (B, C, H, W) feature map
    Output: directions (B, 2, 2, H, W) — two orthogonal unit direction vectors
            step_sizes (B, 2, H, W)    — bounded step sizes [r_min, r_max]
    """

    def __init__(self, in_channels: int, r_min: float = R_MIN, r_max: float = R_MAX):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max

        # Predict 4 values per pixel: 2 for direction + 2 for step sizes
        self.proj = nn.Conv2d(in_channels, 4, kernel_size=1, bias=True)

    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W).

        Returns:
            dirs:  (B, 2, 2, H, W) — dirs[:, 0] is u1, dirs[:, 1] is u2.
                   u1 = [cos(theta), sin(theta)], u2 = [-sin(theta), cos(theta)].
            steps: (B, 2, H, W) — [r1, r2] bounded in [r_min, r_max].
        """
        raw = self.proj(x)  # (B, 4, H, W)

        # ── Direction: normalize (dx, dy) to unit vector ─────────────────
        dx = raw[:, 0:1]  # (B, 1, H, W)
        dy = raw[:, 1:2]  # (B, 1, H, W)
        norm = torch.sqrt(dx ** 2 + dy ** 2 + EPS)
        cos_t = dx / norm  # (B, 1, H, W)
        sin_t = dy / norm  # (B, 1, H, W)

        # u1 = [cos(theta), sin(theta)]  — primary direction
        # u2 = [-sin(theta), cos(theta)] — orthogonal direction (90° rotation)
        u1 = torch.cat([cos_t, sin_t], dim=1)          # (B, 2, H, W)
        u2 = torch.cat([-sin_t, cos_t], dim=1)         # (B, 2, H, W)
        dirs = torch.stack([u1, u2], dim=1)             # (B, 2, 2, H, W)

        # ── Step sizes: sigmoid → [r_min, r_max] ────────────────────────
        r_raw = raw[:, 2:4]                             # (B, 2, H, W)
        steps = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(r_raw)

        return dirs, steps


# ─────────────────────────────────────────────────────────────────────────────
#  Symmetric Bilinear Sampling
#
#  Given a feature map, directions, and step sizes, this function:
#    1. Builds a base coordinate grid (the identity pixel positions).
#    2. For each of the 2 directions, computes forward (+delta) and
#       backward (-delta) offset coordinates.
#    3. Samples the feature map at those 4 continuous locations using
#       PyTorch's grid_sample with bilinear interpolation.
#
#  grid_sample expects coordinates in [-1, +1] normalized space, so we
#  convert pixel offsets accordingly.
#
#  Returns 4 sampled feature maps: (fwd_1, bwd_1, fwd_2, bwd_2) plus the
#  center (original) feature map.
# ─────────────────────────────────────────────────────────────────────────────

def _build_base_grid(B, H, W, device):
    """
    Build a (B, H, W, 2) grid of normalized pixel coordinates in [-1, +1].
    This is the identity grid — sampling with it returns the original image.
    """
    # Create 1D coordinate vectors normalized to [-1, +1]
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)

    # meshgrid returns (H, W) grids; stack into (H, W, 2) with [x, y] order
    # grid_sample expects (x, y) not (y, x)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

    return grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)


def _pixel_offset_to_normalized(offset_px, H, W):
    """
    Convert pixel-space offsets to [-1, +1] normalized offsets for grid_sample.

    grid_sample's [-1, +1] range spans [0, W-1] pixels horizontally and
    [0, H-1] pixels vertically. So 1 pixel = 2/(W-1) in x, 2/(H-1) in y.

    Args:
        offset_px: (B, 2, H, W) pixel offsets [dx_px, dy_px].
        H, W: Spatial dimensions.

    Returns:
        (B, H, W, 2) normalized offsets [dx_norm, dy_norm] for grid_sample.
    """
    # Scale factors: pixels to normalized coordinates
    scale_x = 2.0 / max(W - 1, 1)
    scale_y = 2.0 / max(H - 1, 1)

    dx_norm = offset_px[:, 0] * scale_x  # (B, H, W)
    dy_norm = offset_px[:, 1] * scale_y  # (B, H, W)

    return torch.stack([dx_norm, dy_norm], dim=-1)  # (B, H, W, 2)


def symmetric_sample(feat, dirs, steps):
    """
    Sample 4 symmetric neighbor points from feat using oriented directions.

    For each of the 2 learned orthogonal directions (u1, u2):
      - Forward sample:  feat(p + r_i * u_i)
      - Backward sample: feat(p - r_i * u_i)

    Args:
        feat:  (B, C, H, W) input feature map (can be bottleneck-projected).
        dirs:  (B, 2, 2, H, W) — two orthogonal unit direction vectors.
        steps: (B, 2, H, W) — step sizes [r1, r2].

    Returns:
        center: (B, C, H, W) — the original feature (for reference).
        fwd1, bwd1: (B, C, H, W) — forward/backward along direction 1.
        fwd2, bwd2: (B, C, H, W) — forward/backward along direction 2.
    """
    B, C, H, W = feat.shape

    base_grid = _build_base_grid(B, H, W, feat.device)  # (B, H, W, 2)

    samples = []
    for i in range(2):
        # Direction vector: (B, 2, H, W)
        u = dirs[:, i]                    # (B, 2, H, W)
        r = steps[:, i:i+1]              # (B, 1, H, W)

        # Pixel-space offset: delta = r * u → (B, 2, H, W)
        delta_px = r * u                 # broadcast: (B, 1, H, W) * (B, 2, H, W)

        # Convert to normalized grid offsets
        delta_norm = _pixel_offset_to_normalized(delta_px, H, W)  # (B, H, W, 2)

        # Forward and backward grids
        grid_fwd = base_grid + delta_norm   # (B, H, W, 2)
        grid_bwd = base_grid - delta_norm   # (B, H, W, 2)

        # Bilinear sampling with zero padding for out-of-bounds
        fwd = F.grid_sample(feat, grid_fwd, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        bwd = F.grid_sample(feat, grid_bwd, mode="bilinear",
                            padding_mode="zeros", align_corners=True)

        samples.append((fwd, bwd))

    fwd1, bwd1 = samples[0]
    fwd2, bwd2 = samples[1]
    center = feat

    return center, fwd1, bwd1, fwd2, bwd2


# ─────────────────────────────────────────────────────────────────────────────
#  Differential Token Extraction
#
#  From the 4 symmetric samples + center, we compute physics-inspired tokens:
#
#  1. Context (Even response) — average of forward and backward:
#       a_i = 0.5 * (fwd_i + bwd_i)
#     Captures the local average intensity along each direction.
#
#  2. Gradient (Odd response) — central difference:
#       g_i = (fwd_i - bwd_i) / (2 * r_i + eps)
#     Captures the rate of change (edge strength) along each direction.
#     These directional gradients are rotated back to image-axis-aligned
#     Gx and Gy using the known direction vectors u1, u2.
#
#  3. Curvature / Laplacian (Even response) — second-order central difference:
#       s_i = (fwd_i - 2*center + bwd_i) / (r_i^2 + eps)
#       L   = 0.5 * (s1 + s2)
#     Captures whether the pixel is on a ridge, valley, or flat area.
#
#  These tokens are concatenated along the channel dimension and fed into
#  the 1×1 mixing layer.
# ─────────────────────────────────────────────────────────────────────────────

def extract_differential_tokens(center, fwd1, bwd1, fwd2, bwd2, dirs, steps):
    """
    Compute context, gradient, and curvature tokens from symmetric samples.

    Args:
        center: (B, C, H, W)
        fwd1, bwd1, fwd2, bwd2: (B, C, H, W) symmetric samples.
        dirs:  (B, 2, 2, H, W) orthogonal direction vectors.
        steps: (B, 2, H, W) step sizes.

    Returns:
        tokens: (B, 6*C, H, W) — concatenation of
                [center, a1, a2, Gx, Gy, L].
    """
    # Step sizes for broadcasting with (B, C, H, W)
    r1 = steps[:, 0:1]  # (B, 1, H, W)
    r2 = steps[:, 1:2]  # (B, 1, H, W)

    # ── Context tokens (even response): local directional averages ───────
    a1 = 0.5 * (fwd1 + bwd1)  # (B, C, H, W)
    a2 = 0.5 * (fwd2 + bwd2)  # (B, C, H, W)

    # ── Directional gradients (odd response): central differences ────────
    # g1 along direction u1, g2 along direction u2
    g1 = (fwd1 - bwd1) / (2.0 * r1 + EPS)  # (B, C, H, W)
    g2 = (fwd2 - bwd2) / (2.0 * r2 + EPS)  # (B, C, H, W)

    # Rotate directional gradients to image-axis-aligned Gx, Gy.
    # u1 = [cos_t, sin_t], u2 = [-sin_t, cos_t]
    # Gx = g1 * cos_t + g2 * (-sin_t)
    # Gy = g1 * sin_t + g2 * cos_t
    cos_t = dirs[:, 0, 0:1]  # (B, 1, H, W) — cos(theta) from u1[0]
    sin_t = dirs[:, 0, 1:2]  # (B, 1, H, W) — sin(theta) from u1[1]

    Gx = g1 * cos_t + g2 * (-sin_t)  # (B, C, H, W)
    Gy = g1 * sin_t + g2 * cos_t     # (B, C, H, W)

    # ── Curvature / Laplacian (even response): second-order differences ──
    s1 = (fwd1 - 2.0 * center + bwd1) / (r1 ** 2 + EPS)  # (B, C, H, W)
    s2 = (fwd2 - 2.0 * center + bwd2) / (r2 ** 2 + EPS)  # (B, C, H, W)
    L  = 0.5 * (s1 + s2)                                  # (B, C, H, W)

    # ── Concatenate all tokens ───────────────────────────────────────────
    # 6 tokens × C channels each = 6C total channels
    tokens = torch.cat([center, a1, a2, Gx, Gy, L], dim=1)  # (B, 6*C, H, W)

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
#  Token Mixer
#
#  Takes the 6*inner-channel token tensor and projects it to out_channels
#  using two lightweight 1×1 convolutions:
#    - First 1×1: mixes the 6 token types (context, gradient, curvature)
#      from 6*inner channels down to out_channels.
#    - Second 1×1: refines the mixed representation at out_channels.
#
#  This replaces the heavy 3×3 spatial convolutions in standard U-Net.
#  Because all spatial geometry is already encoded in the token values
#  via the oriented sampling, only channel-wise mixing is needed.
#
#  Previous version used a full C→C gate (C*C parameters) which caused
#  the parameter explosion. This version is strictly 1×1 projection only.
# ─────────────────────────────────────────────────────────────────────────────

class TokenMixer(nn.Module):
    """
    Two-stage 1×1 projection for differential tokens.

    Stage 1: Mix 6 token types → out_channels (token fusion).
    Stage 2: Refine at out_channels (feature polishing).

    Input:  (B, 6 * inner, H, W) — concatenated differential tokens.
    Output: (B, out_channels, H, W) — mixed feature map.
    """

    def __init__(self, token_channels: int, out_channels: int):
        """
        Args:
            token_channels: Number of input channels (should be 6 * inner).
            out_channels:   Number of output channels.
        """
        super().__init__()

        self.mix = nn.Sequential(
            # Stage 1: fuse the 6 differential token types
            nn.Conv2d(token_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Stage 2: refine the fused features
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens):
        return self.mix(tokens)


# ─────────────────────────────────────────────────────────────────────────────
#  GeoSample2D: Stride-1 Refinement Block
#
#  This is the core operator that REPLACES the standard DoubleConv block.
#
#  Architecture:
#    1. GeometryHead predicts local orientation + step sizes from FULL features.
#       (Cheap: C_in * 4 params. Sees all channel information.)
#    2. Bottleneck projection compresses features from C_in to inner dim.
#       (Prevents the 6x token expansion from exploding at high channel counts.)
#    3. Symmetric sampling probes 4 oriented neighbors on the PROJECTED features.
#       (Zero learnable params — pure bilinear interpolation.)
#    4. Differential tokens are extracted (context, gradient, curvature).
#       (Zero learnable params — pure arithmetic.)
#    5. TokenMixer projects 6*inner tokens to out_channels via two 1×1 convs.
#    6. Residual connection: output = update + identity (when shapes match,
#       otherwise a 1×1 projection adapts the skip path).
#
#  Parameter budget example at bottleneck level (256 → 512):
#    - Geometry head:     256 * 4 + 4  =     1,028
#    - Bottleneck proj:   256 * 64     =    16,384
#    - TokenMixer:   384 * 512 + 512 * 512 = 458,752
#    - Residual:          256 * 512    =   131,072
#    Total: ~607K  vs  standard DoubleConv: 256*256*9 + 256*512*9 ≈ 1.77M
# ─────────────────────────────────────────────────────────────────────────────

class GeoSample2D(nn.Module):
    """
    Geometry-guided stride-1 refinement block.

    Replaces DoubleConv: instead of two 3×3 convolutions, it performs
    oriented sampling on bottleneck-projected features, extracts differential
    tokens, and mixes with 1×1 convolutions.

    Input:  (B, C_in,  H, W)
    Output: (B, C_out, H, W) — same spatial size, refined features.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        inner = _compute_inner_dim(in_channels)
        self.inner = inner

        # ── Geometry prediction from FULL features (always cheap) ────────
        self.geo_head = GeometryHead2D(in_channels)

        # ── Bottleneck: compress features for efficient sampling ─────────
        self.proj_down = nn.Sequential(
            nn.Conv2d(in_channels, inner, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )

        # ── Token mixer: 6 * inner → out_channels ───────────────────────
        self.mixer = TokenMixer(6 * inner, out_channels)

        # ── Residual projection if channel counts differ ─────────────────
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.residual(x)

        # 1. Predict geometry from FULL features
        dirs, steps = self.geo_head(x)

        # 2. Project to bottleneck dimension for sampling
        x_inner = self.proj_down(x)  # (B, inner, H, W)

        # 3. Sample 4 oriented neighbors on the compressed features
        center, fwd1, bwd1, fwd2, bwd2 = symmetric_sample(
            x_inner, dirs, steps
        )

        # 4. Extract differential tokens (context, gradient, curvature)
        tokens = extract_differential_tokens(
            center, fwd1, bwd1, fwd2, bwd2, dirs, steps
        )  # (B, 6 * inner, H, W)

        # 5. Mix tokens to output channels
        out = self.mixer(tokens)  # (B, C_out, H, W)

        # 6. Residual addition
        return out + identity


# ─────────────────────────────────────────────────────────────────────────────
#  GeoDownsample2D: Stride-2 Downsampling Block
#
#  REPLACES MaxPool2d(2) with a dual-path downsampler that preserves
#  directional edge energy during spatial reduction.
#
#  Why MaxPool fails at boundaries:
#    MaxPool keeps only the highest value in each 2×2 patch. If a boundary
#    produces a positive gradient on one side and negative on the other,
#    one gets discarded. Edge information is permanently lost.
#
#  Dual-path solution:
#    Path A — AvgPool2d: Safely captures low-frequency background context
#             by averaging. Stable and smooth.
#    Path B — Directional Compensation: Computes gradient magnitudes
#             |Gx|, |Gy| BEFORE pooling (using absolute values so opposing
#             edges don't cancel). These are average-pooled separately and
#             projected via 1×1 conv as a correction term.
#
#  Final output = Path A + Path B (both at half resolution).
#
#  Uses the same bottleneck projection for gradient computation to keep
#  memory and parameter counts low.
# ─────────────────────────────────────────────────────────────────────────────

class GeoDownsample2D(nn.Module):
    """
    Geometry-aware stride-2 downsampler.

    Replaces MaxPool2d: preserves directional edge cues during spatial reduction.

    Input:  (B, C, H,   W)
    Output: (B, C, H/2, W/2)
    """

    def __init__(self, channels: int):
        super().__init__()

        inner = _compute_inner_dim(channels)

        # Path A: standard average pooling (low-frequency path)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Geometry head to extract directional gradients before pooling
        self.geo_head = GeometryHead2D(channels)

        # Bottleneck projection for gradient computation
        self.proj_down = nn.Sequential(
            nn.Conv2d(channels, inner, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )

        # Path B: project |Gx|, |Gy|, |L| compensation (3 * inner → channels)
        self.compensation = nn.Sequential(
            nn.Conv2d(inner * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # ── Path A: Average pooling (stable low-frequency) ──────────────
        low_freq = self.avg_pool(x)  # (B, C, H/2, W/2)

        # ── Path B: Directional compensation ─────────────────────────────
        # Predict geometry from full features
        dirs, steps = self.geo_head(x)

        # Project to bottleneck for efficient sampling
        x_inner = self.proj_down(x)  # (B, inner, H, W)

        # Sample and compute differentials on bottleneck features
        center, fwd1, bwd1, fwd2, bwd2 = symmetric_sample(
            x_inner, dirs, steps
        )

        # Compute gradient magnitudes (absolute to prevent cancellation)
        r1 = steps[:, 0:1]  # (B, 1, H, W)
        r2 = steps[:, 1:2]

        g1 = (fwd1 - bwd1) / (2.0 * r1 + EPS)
        g2 = (fwd2 - bwd2) / (2.0 * r2 + EPS)

        # Rotate to image axes
        cos_t = dirs[:, 0, 0:1]
        sin_t = dirs[:, 0, 1:2]
        Gx = g1 * cos_t + g2 * (-sin_t)
        Gy = g1 * sin_t + g2 * cos_t

        # Curvature
        s1 = (fwd1 - 2.0 * center + bwd1) / (r1 ** 2 + EPS)
        s2 = (fwd2 - 2.0 * center + bwd2) / (r2 ** 2 + EPS)
        L  = 0.5 * (s1 + s2)

        # Take absolute values so opposing edges don't cancel during pooling
        high_freq = torch.cat(
            [Gx.abs(), Gy.abs(), L.abs()], dim=1
        )  # (B, 3 * inner, H, W)

        # Pool the magnitudes and project back to full channel count
        high_freq = self.avg_pool(high_freq)             # (B, 3*inner, H/2, W/2)
        compensation = self.compensation(high_freq)      # (B, C, H/2, W/2)

        # ── Combine both paths ───────────────────────────────────────────
        return low_freq + compensation


# ─────────────────────────────────────────────────────────────────────────────
#  ConsensusSkip2D: Skip-Connection Geometric Alignment
#
#  REPLACES naive torch.cat([upsampled_decoder, encoder_skip]).
#
#  Problem: When the decoder upsamples from lower resolution, its boundary
#  estimates are slightly blurry and can be spatially misaligned with the
#  crisp encoder skip features. Naive concatenation preserves this mismatch,
#  causing ghosting artifacts at tumor edges.
#
#  Solution:
#    1. Predict local geometry (direction + step size) from both encoder skip
#       features and decoder features independently.
#    2. Compute a direction agreement score (cosine similarity between the
#       two predicted direction vectors).
#    3. Use the agreement score as a gate to blend the geometries.
#    4. Refine the decoder features using the consensus geometry before
#       concatenating with skip features.
#
#  This ensures that when encoder and decoder agree on boundary orientation,
#  the concatenated features are geometrically coherent.
# ─────────────────────────────────────────────────────────────────────────────

class ConsensusSkip2D(nn.Module):
    """
    Geometry-aligned skip connection.

    Takes encoder skip features and upsampled decoder features, aligns their
    geometric representations, and returns their concatenation.

    Input:  skip_enc (B, C_enc, H, W), feat_dec (B, C_dec, H, W)
    Output: (B, C_enc + C_dec, H, W) — aligned concatenation.
    """

    def __init__(self, enc_channels: int, dec_channels: int):
        super().__init__()

        # Predict geometry from encoder and decoder independently
        self.geo_enc = GeometryHead2D(enc_channels)
        self.geo_dec = GeometryHead2D(dec_channels)

        # Lightweight refinement of decoder features using consensus geometry
        self.refine = nn.Sequential(
            nn.Conv2d(dec_channels, dec_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_dec, skip_enc):
        """
        Args:
            feat_dec: Upsampled decoder features (B, C_dec, H, W).
            skip_enc: Encoder skip features (B, C_enc, H, W).

        Returns:
            Concatenated, geometry-aligned features (B, C_enc + C_dec, H, W).
        """
        # Handle size mismatch (in case of odd dimensions)
        if feat_dec.shape[2:] != skip_enc.shape[2:]:
            feat_dec = F.interpolate(
                feat_dec, size=skip_enc.shape[2:],
                mode="bilinear", align_corners=True
            )

        # Predict directions from both branches
        dirs_enc, _ = self.geo_enc(skip_enc)   # (B, 2, 2, H, W)
        dirs_dec, _ = self.geo_dec(feat_dec)   # (B, 2, 2, H, W)

        # Cosine similarity between primary directions (u1)
        # u1_enc: (B, 2, H, W), u1_dec: (B, 2, H, W)
        u1_enc = dirs_enc[:, 0]  # (B, 2, H, W)
        u1_dec = dirs_dec[:, 0]  # (B, 2, H, W)

        # Dot product along the direction dimension
        # |cos(angle_diff)| — absolute because opposite directions are equivalent
        agreement = (u1_enc * u1_dec).sum(dim=1, keepdim=True).abs()  # (B, 1, H, W)
        # Clamp to [0, 1] for safety
        agreement = agreement.clamp(0.0, 1.0)

        # Use agreement as a gate to refine decoder features
        # High agreement → trust the decoder boundary placement
        # Low agreement → dampen the decoder signal in that region
        feat_dec_refined = feat_dec + agreement * self.refine(feat_dec)

        # Concatenate along channels
        return torch.cat([skip_enc, feat_dec_refined], dim=1)
