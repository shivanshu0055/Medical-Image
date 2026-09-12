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
#  Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def build_unet(config: dict | None = None) -> UNet:
    """
    Build a U-Net model using settings from config.yaml.

    Args:
        config: The 'segmentation' section of config.yaml.
                If None, uses default values.

    Returns:
        UNet model (not yet trained, weights are random).
    """
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
