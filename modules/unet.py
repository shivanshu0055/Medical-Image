"""
MedBoard — modules/unet.py
Phase 2: U-Net Segmentation Model

U-Net is a neural network designed specifically for medical image segmentation.
It takes a brain MRI image and outputs a mask showing WHERE the tumor is,
pixel by pixel.

Architecture overview:
  - Encoder (left side of U): reads the image and extracts features, getting
    smaller and smaller (like zooming out to understand context).
  - Bottleneck (bottom of U): the most compressed representation.
  - Decoder (right side of U): builds the mask back up to original size.
  - Skip connections: directly connect each encoder level to the matching
    decoder level, so fine details aren't lost during upsampling.

Input:  (batch, 3, 256, 256)  — RGB MRI image
Output: (batch, 1, 256, 256)  — probability map (0=background, 1=tumor)
                                 Apply sigmoid + threshold 0.5 to get binary mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Building Block: Double Convolution
#
#  Almost every block in U-Net does the same thing twice:
#    Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
#
#  Why twice? One convolution captures basic patterns, the second one
#  refines them. This is the standard U-Net recipe.
#
#  BatchNorm: normalizes the outputs of each layer so training is stable.
#  ReLU: the activation function — replaces negative values with 0,
#        which helps the network learn non-linear patterns.
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels:  Number of channels coming in (e.g. 3 for RGB image).
            out_channels: Number of feature maps to produce.
        """
        super().__init__()

        self.block = nn.Sequential(
            # First convolution
            # kernel_size=3: looks at a 3×3 region around each pixel
            # padding=1: adds a 1-pixel border so output stays same size as input
            # bias=False: BatchNorm handles the bias, so we skip it here
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),   # stabilize outputs
            nn.ReLU(inplace=True),           # apply non-linearity

            # Second convolution (same settings)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder Block (going DOWN the U)
#
#  Each encoder step:
#    1. Apply DoubleConv to extract features at this scale.
#    2. Save the output (we'll use it again in the decoder via skip connection).
#    3. MaxPool: shrink the spatial size by half (e.g. 256×256 → 128×128).
#
#  MaxPool keeps the most "active" pixel in each 2×2 region.
#  It's like summarizing — you lose fine detail but gain broader understanding.
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """One step of the encoder: DoubleConv → save features → MaxPool."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H and W

    def forward(self, x):
        features = self.conv(x)    # extract features at this resolution
        pooled   = self.pool(features)  # shrink for the next encoder step
        return features, pooled    # return BOTH: features for skip, pooled to go deeper


# ─────────────────────────────────────────────────────────────────────────────
#  Decoder Block (going UP the U)
#
#  Each decoder step:
#    1. Upsample: double the spatial size (e.g. 16×16 → 32×32).
#    2. Concatenate: glue the skip connection features from the encoder
#       onto the upsampled output. This is how the decoder "remembers"
#       the fine details from earlier.
#    3. Apply DoubleConv to combine and refine.
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """One step of the decoder: Upsample → Concatenate skip → DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Bilinear upsampling: smoothly doubles the spatial size.
        # More stable than transposed convolution (avoids checkerboard artifacts).
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # After concatenating, channel count doubles (skip + upsampled),
        # so in_channels here accounts for that.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        """
        Args:
            x:    The feature map coming from the previous decoder step (or bottleneck).
            skip: The feature map saved from the matching encoder step.
        """
        x = self.upsample(x)    # double the size

        # Handle edge case: if sizes don't match perfectly due to odd input dims
        # (shouldn't happen with 256×256 but good practice)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        # Concatenate along the channel dimension
        # e.g. (B, 128, 64, 64) + (B, 128, 64, 64) → (B, 256, 64, 64)
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)     # combine and refine


# ─────────────────────────────────────────────────────────────────────────────
#  Attention Gate (Additive Attention — Oktay et al., 2018)
#
#  Filters skip connection features using a gating signal from the deeper
#  decoder layer. Irrelevant background features are suppressed (assigned weights
#  near 0), while salient tumor region features are preserved (weights near 1).
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Additive Attention Gate (Oktay et al., 2018: 'Attention U-Net').

    Args:
        gate_channels:  Number of channels in gating signal g (from decoder level below).
        skip_channels:  Number of channels in skip connection x (from encoder).
        inter_channels: Intermediate projection channels (defaults to skip_channels // 2).
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int | None = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)

        # W_g: projects gating signal g to inter_channels
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # W_x: projects encoder skip features x to inter_channels
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # psi: collapses to 1-channel spatial attention coefficient map
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder level below (B, gate_channels, H_g, W_g).
            x: Skip connection features from encoder (B, skip_channels, H_x, W_x).

        Returns:
            Attention-weighted skip tensor (B, skip_channels, H_x, W_x).
        """
        # Align spatial dimensions of g to match x if different
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        net = self.relu(g1 + x1)
        alpha = self.psi(net)   # (B, 1, H_x, W_x) in [0, 1]
        self.last_alpha = alpha # cached for attention map visualization

        return x * alpha


class AttentionDecoderBlock(nn.Module):
    """
    One step of the Attention U-Net decoder:
      1. Upsample decoder feature map x from previous step / bottleneck.
      2. Filter encoder skip features using AttentionGate(g=x, x=skip).
      3. Concatenate gated skip and upsampled decoder features along channels.
      4. DoubleConv to combine and refine.
    """

    def __init__(self, gate_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attn = AttentionGate(
            gate_channels=gate_channels,
            skip_channels=skip_channels,
            inter_channels=max(skip_channels // 2, 1),
        )
        self.conv = DoubleConv(gate_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    Decoder features from previous stage or bottleneck (B, gate_channels, H, W).
            skip: Encoder skip features (B, skip_channels, 2H, 2W).
        """
        x = self.upsample(x)  # double spatial size

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        # Gate the skip connection before concatenation
        gated_skip = self.attn(g=x, x=skip)

        # Concatenate along channel dimension
        x = torch.cat([gated_skip, x], dim=1)

        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
#  The Full U-Net
#
#  With base_features=32, the channel sizes look like this:
#
#  Encoder:
#    Input     (3, 256, 256)
#    Enc1 out  (32,  256, 256) → pooled: (32,  128, 128)
#    Enc2 out  (64,  128, 128) → pooled: (64,   64,  64)
#    Enc3 out  (128,  64,  64) → pooled: (128,  32,  32)
#    Enc4 out  (256,  32,  32) → pooled: (256,  16,  16)
#
#  Bottleneck:
#    (512, 16, 16)
#
#  Decoder (skip channels + upsampled channels → out channels):
#    Dec4: (512+256=768 → 256) at (32, 32)
#    Dec3: (256+128=384 → 128) at (64, 64)
#    Dec2: (128+64=192  → 64)  at (128, 128)
#    Dec1: (64+32=96    → 32)  at (256, 256)
#
#  Output:
#    1×1 Conv: (32 → 1) → single channel probability map
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for binary brain tumor segmentation.

    Takes an RGB MRI image, outputs a single-channel tumor probability map.
    Apply sigmoid + threshold 0.5 to convert to a binary mask.

    Args:
        in_channels:   Number of input channels (3 for RGB).
        out_channels:  Number of output channels (1 for binary mask).
        base_features: Starting number of feature maps. Doubles at each encoder
                       level. Default 32 → sizes: 32, 64, 128, 256, 512.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_features: int = 32):
        super().__init__()

        f = base_features  # shorthand — f=32 means 32, 64, 128, 256, 512

        # ── Encoder (going down) ──────────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels, f)       # 3   → 32
        self.enc2 = EncoderBlock(f,           f * 2)   # 32  → 64
        self.enc3 = EncoderBlock(f * 2,       f * 4)   # 64  → 128
        self.enc4 = EncoderBlock(f * 4,       f * 8)   # 128 → 256

        # ── Bottleneck (bottom of the U) ──────────────────────────────────
        # No pooling here — this is the most compressed representation.
        # Captures the high-level meaning of the image.
        self.bottleneck = DoubleConv(f * 8, f * 16)   # 256 → 512

        # ── Decoder (going up) ────────────────────────────────────────────
        # Each decoder block receives: upsampled from below + skip from encoder.
        # Combined channels = (channels from below) + (skip channels)
        self.dec4 = DecoderBlock(f * 16 + f * 8,  f * 8)   # 512+256=768 → 256
        self.dec3 = DecoderBlock(f * 8  + f * 4,  f * 4)   # 256+128=384 → 128
        self.dec2 = DecoderBlock(f * 4  + f * 2,  f * 2)   # 128+64=192  → 64
        self.dec1 = DecoderBlock(f * 2  + f,       f)       # 64+32=96    → 32

        # ── Output Layer ──────────────────────────────────────────────────
        # 1×1 convolution: maps 32 feature maps down to 1 output channel.
        # This single channel is the tumor probability for each pixel.
        # NOTE: No sigmoid here — we apply it separately so we can use
        # BCEWithLogitsLoss during training (numerically more stable).
        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Args:
            x: Input tensor of shape (batch, 3, 256, 256).

        Returns:
            Logits tensor of shape (batch, 1, 256, 256).
            Apply sigmoid to get probabilities, then threshold at 0.5 for mask.
        """
        # ── Encoder ──
        skip1, x = self.enc1(x)   # skip1: (B, 32,  256, 256) | x: (B, 32,  128, 128)
        skip2, x = self.enc2(x)   # skip2: (B, 64,  128, 128) | x: (B, 64,   64,  64)
        skip3, x = self.enc3(x)   # skip3: (B, 128,  64,  64) | x: (B, 128,  32,  32)
        skip4, x = self.enc4(x)   # skip4: (B, 256,  32,  32) | x: (B, 256,  16,  16)

        # ── Bottleneck ──
        x = self.bottleneck(x)    # x: (B, 512, 16, 16)

        # ── Decoder (skip connections pass encoder features back in) ──
        x = self.dec4(x, skip4)   # x: (B, 256, 32,  32)
        x = self.dec3(x, skip3)   # x: (B, 128, 64,  64)
        x = self.dec2(x, skip2)   # x: (B, 64,  128, 128)
        x = self.dec1(x, skip1)   # x: (B, 32,  256, 256)

        # ── Output ──
        return self.output_conv(x)  # (B, 1, 256, 256) — raw logits


# ─────────────────────────────────────────────────────────────────────────────
#  Attention U-Net (Oktay et al., 2018)
#
#  Incorporates Attention Gates onto each skip connection.
#  Exact same encoder, bottleneck, and channel progression as standard U-Net,
#  but each skip connection is dynamically weighted by the decoder gating signal.
# ─────────────────────────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary brain tumor segmentation.

    Uses Attention Gates to suppress irrelevant background regions and
    highlight tumor features before concatenating skip connections.

    Args:
        in_channels:   Number of input channels (3 for RGB).
        out_channels:  Number of output channels (1 for binary mask).
        base_features: Starting number of feature maps. Doubles at each encoder level.
                       Default 32 → sizes: 32, 64, 128, 256, 512.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_features: int = 32):
        super().__init__()

        f = base_features

        # ── Encoder (same as standard U-Net) ──────────────────────────────
        self.enc1 = EncoderBlock(in_channels, f)       # 3   → 32
        self.enc2 = EncoderBlock(f,           f * 2)   # 32  → 64
        self.enc3 = EncoderBlock(f * 2,       f * 4)   # 64  → 128
        self.enc4 = EncoderBlock(f * 4,       f * 8)   # 128 → 256

        # ── Bottleneck (bottom of the U) ──────────────────────────────────
        self.bottleneck = DoubleConv(f * 8, f * 16)   # 256 → 512

        # ── Decoder with Attention Gates (going up) ────────────────────────
        # dec4: gating signal from bottleneck (512), skip from enc4 (256) -> out (256)
        self.dec4 = AttentionDecoderBlock(gate_channels=f * 16, skip_channels=f * 8, out_channels=f * 8)
        # dec3: gating signal from dec4 (256), skip from enc3 (128) -> out (128)
        self.dec3 = AttentionDecoderBlock(gate_channels=f * 8,  skip_channels=f * 4, out_channels=f * 4)
        # dec2: gating signal from dec3 (128), skip from enc2 (64) -> out (64)
        self.dec2 = AttentionDecoderBlock(gate_channels=f * 4,  skip_channels=f * 2, out_channels=f * 2)
        # dec1: gating signal from dec2 (64), skip from enc1 (32) -> out (32)
        self.dec1 = AttentionDecoderBlock(gate_channels=f * 2,  skip_channels=f,     out_channels=f)

        # ── Output Layer ──────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Attention U-Net.

        Args:
            x: Input tensor of shape (batch, 3, 256, 256).

        Returns:
            Logits tensor of shape (batch, 1, 256, 256).
        """
        # ── Encoder ──
        skip1, x = self.enc1(x)   # skip1: (B, 32,  256, 256)
        skip2, x = self.enc2(x)   # skip2: (B, 64,  128, 128)
        skip3, x = self.enc3(x)   # skip3: (B, 128,  64,  64)
        skip4, x = self.enc4(x)   # skip4: (B, 256,  32,  32)

        # ── Bottleneck ──
        x = self.bottleneck(x)    # x: (B, 512, 16, 16)

        # ── Attention-Gated Decoder ──
        x = self.dec4(x, skip4)   # x: (B, 256, 32,  32)
        x = self.dec3(x, skip3)   # x: (B, 128, 64,  64)
        x = self.dec2(x, skip2)   # x: (B, 64,  128, 128)
        x = self.dec1(x, skip1)   # x: (B, 32,  256, 256)

        # ── Output ──
        return self.output_conv(x)  # (B, 1, 256, 256) — raw logits


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def build_attention_unet(config: dict | None = None) -> AttentionUNet:
    """
    Build an Attention U-Net model using settings from config.yaml.

    Args:
        config: The 'segmentation' section of config.yaml.
                If None, uses default values.

    Returns:
        AttentionUNet model (not yet trained, weights are random).
    """
    if config is None:
        return AttentionUNet(in_channels=3, out_channels=1, base_features=32)

    return AttentionUNet(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 1),
        base_features=config.get("base_features", 32),
    )


def build_unet(config: dict | None = None) -> UNet | AttentionUNet:
    """
    Build a U-Net model using settings from config.yaml.
    If config['architecture'] == 'attention_unet', builds an AttentionUNet.

    Args:
        config: The 'segmentation' section of config.yaml.
                If None, uses default values.

    Returns:
        UNet or AttentionUNet model (not yet trained, weights are random).
    """
    if config is not None and config.get("architecture", "").lower() == "attention_unet":
        return build_attention_unet(config)

    if config is None:
        return UNet(in_channels=3, out_channels=1, base_features=32)

    return UNet(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 1),
        base_features=config.get("base_features", 32),
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_mask(model: nn.Module, image_tensor: torch.Tensor,
                 threshold: float = 0.5, device: str = "cpu") -> torch.Tensor:
    """
    Run inference and return a binary mask.

    Args:
        model:        Trained UNet model.
        image_tensor: Preprocessed image tensor (3, 256, 256) or (B, 3, 256, 256).
        threshold:    Probability cutoff for tumor vs background. Default 0.5.
        device:       'cuda' or 'cpu'.

    Returns:
        Binary mask tensor of shape (1, 256, 256) or (B, 1, 256, 256),
        with values exactly 0.0 (background) or 1.0 (tumor).
    """
    model.eval()  # turn off dropout/batchnorm training behaviour

    # Add batch dimension if a single image was passed
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)   # (3,H,W) → (1,3,H,W)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():          # don't compute gradients (saves memory)
        logits = model(image_tensor)           # raw output (B, 1, H, W)
        probs  = torch.sigmoid(logits)         # convert to probabilities [0, 1]
        mask   = (probs > threshold).float()   # threshold → binary {0.0, 1.0}

    return mask.squeeze(0) if mask.shape[0] == 1 else mask  # remove batch dim if single
