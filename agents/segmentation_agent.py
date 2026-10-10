"""
MedBoard — agents/segmentation_agent.py
Phase 2: Segmentation Agent (LangGraph Node)

This file wraps the U-Net model as a LangGraph agent node.

What does the agent do?
  - Receives the MedBoardState (the shared data object passed between all agents)
  - Loads the trained U-Net
  - Runs inference on the uploaded MRI image
  - Fills in tumor mask, tumor presence flag, and confidence score
  - Passes the updated state to the next agent

In the LangGraph pipeline, this node is called "segmentation_agent".
It runs AFTER preprocessing and BEFORE classification.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from modules.unet import build_unet, predict_mask
from modules.preprocessing import MRIPreprocessor


# ─────────────────────────────────────────────────────────────────────────────
#  Segmentation Agent Node
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationAgent:
    """
    LangGraph node that runs U-Net inference on an MRI image.

    Usage in the pipeline:
        agent = SegmentationAgent()
        state = agent.run(state)   # state.tumor_mask is now filled in

    The agent is lazy-loaded — the model is loaded from disk only on the first
    call to run(), not when the object is created. This keeps startup fast.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.seg_cfg  = cfg["segmentation"]
        self.train_cfg = cfg["training"]

        self.weights_path = self.seg_cfg["weights_path"]
        self.threshold    = self.seg_cfg["mask_threshold"]   # 0.5
        self.device_str   = self.train_cfg["device"]          # "cuda" or "cpu"

        self.preprocessor = MRIPreprocessor(config_path)

        # Model loaded lazily on first inference call
        self._model = None

    def _load_model(self):
        """Load the trained U-Net weights from disk (called once, on first use)."""
        weights_path = Path(self.weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Segmentation model weights not found at '{weights_path}'.\n"
                f"Please train the model first using notebooks/train_segmentation.ipynb."
            )

        device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )

        # Detect architecture and base_features from checkpoint
        has_geo_keys = any("geo_conv" in k for k in state_dict.keys())
        has_attn_keys = any("attn.W_g" in k or "attn.psi" in k for k in state_dict.keys())
        has_unet_keys = any("conv.block" in k for k in state_dict.keys()) and not has_attn_keys

        if has_geo_keys:
            arch_type = "geosample_unet"
        elif has_attn_keys:
            arch_type = "attention_unet"
        elif has_unet_keys:
            arch_type = "unet"
        else:
            cfg_arch = self.seg_cfg.get("architecture", "").lower()
            if "attention" in cfg_arch or "attention" in weights_path.name.lower():
                arch_type = "attention_unet"
            elif "geosample" in cfg_arch or "geosample" in weights_path.name.lower():
                arch_type = "geosample_unet"
            else:
                arch_type = "unet"

        model_cfg = dict(self.seg_cfg)
        if "output_conv.weight" in state_dict:
            model_cfg["base_features"] = state_dict["output_conv.weight"].shape[1]

        if arch_type == "geosample_unet":
            from modules.geosample_unet import build_geosample_unet
            model = build_geosample_unet(model_cfg)
            arch_name = f"GeoSampleUNet (base_features={model_cfg.get('base_features')})"
        elif arch_type == "attention_unet":
            from modules.unet import build_attention_unet
            model = build_attention_unet(model_cfg)
            arch_name = f"Attention UNet (base_features={model_cfg.get('base_features')})"
        else:
            model = build_unet(model_cfg)
            arch_name = f"Standard UNet (base_features={model_cfg.get('base_features')})"

        # strict=False allows loading checkpoints trained with auxiliary heads (e.g. Run 3)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[SegmentationAgent] Note: missing keys: {missing_keys}")
        model = model.to(device)
        model.eval()   # inference mode

        self._model = model
        self._device = device
        print(f"[SegmentationAgent] {arch_name} loaded from {weights_path} on {device}")

    def run(self, state: dict) -> dict:
        """
        LangGraph node function — runs U-Net segmentation on the uploaded image.

        Args:
            state: MedBoardState dict. Must have 'image_path' filled in.

        Returns:
            Updated state with:
              - state['tumor_mask']:              (1, 256, 256) binary tensor
              - state['has_tumor']:               True/False
              - state['segmentation_confidence']: float, fraction of image marked as tumor
              - state['seg_error']:               error message if something went wrong
        """
        # Lazy-load the model on first call
        if self._model is None:
            try:
                self._load_model()
            except FileNotFoundError as e:
                state["seg_error"] = str(e)
                state["has_tumor"] = None
                return state

        image_path = state.get("image_path")
        if not image_path:
            state["seg_error"] = "No image_path provided in state."
            return state

        try:
            # Preprocess image for segmentation model
            preprocessed = self.preprocessor.preprocess_image(
                image_path, mode="segmentation"
            )
            image_tensor = preprocessed.tensor   # (3, 256, 256)

            # Run U-Net inference
            mask = predict_mask(
                self._model,
                image_tensor,
                threshold=self.threshold,
                device=str(self._device),
            )   # shape: (1, 256, 256), values: {0.0, 1.0}

            # Tumor coverage: what fraction of the image is marked as tumor
            tumor_pixels  = mask.sum().item()
            total_pixels  = mask.numel()
            tumor_fraction = tumor_pixels / total_pixels

            # Update state
            state["tumor_mask"]              = mask
            state["has_tumor"]               = tumor_fraction > 0.001  # >0.1% of image
            state["segmentation_confidence"] = round(tumor_fraction * 100, 2)  # as percentage
            state["seg_error"]               = None

        except Exception as e:
            state["seg_error"] = f"Segmentation failed: {str(e)}"
            state["has_tumor"] = None
            state["tumor_mask"] = None
            state["segmentation_confidence"] = 0.0

        return state


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraph Node Function
#  (LangGraph expects a plain function, not a class method)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level agent instance (created once, reused across pipeline runs)
_agent_instance: SegmentationAgent | None = None


def segmentation_node(state: dict) -> dict:
    """
    LangGraph-compatible node function for the segmentation step.

    Wire this into your LangGraph graph like:
        graph.add_node("segmentation", segmentation_node)

    Args:
        state: The shared MedBoardState dictionary.

    Returns:
        Updated state dictionary.
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SegmentationAgent()
    return _agent_instance.run(state)
