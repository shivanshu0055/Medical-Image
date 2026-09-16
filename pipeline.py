"""
MedBoard — pipeline.py
End-to-end pipeline orchestrator for MedBoard agents.

Executes the available pipeline stages on a single MRI image:
  1. Preprocessing (validation and normalization)
  2. Segmentation Agent (U-Net tumor mask localization)
  3. Classification Agent (EfficientNet-B0 tumor typing)
  (Phases 4-6 will chain Feature Extraction, RAG, and LLM Synthesis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np


# ─── Shared State Schema ──────────────────────────────────────
@dataclass
class MedBoardState:
    """
    Shared state passed between all LangGraph agents.
    Implements dict-like item access (__getitem__, __setitem__, get)
    for seamless compatibility across LangGraph and custom agents.
    """

    # Input
    image_path: str = ""

    # Phase 1 — Preprocessing output
    preprocessed_image: Optional[np.ndarray] = None   # (C, H, W) float32

    # Phase 2 — Segmentation output
    tumor_mask: Optional[np.ndarray] = None            # (H, W) or (1, H, W) binary
    has_tumor: bool = False
    segmentation_confidence: float = 0.0
    seg_error: Optional[str] = None

    # Phase 3 — Classification output
    predicted_class: str = ""
    classification_confidence: float = 0.0
    class_probabilities: dict[str, float] = field(default_factory=dict)
    clf_error: Optional[str] = None

    # Phase 4 — Feature extraction output
    tumor_features: dict[str, Any] = field(default_factory=dict)

    # Phase 5 — Evidence / RAG output
    evidence_snippets: list[dict[str, Any]] = field(default_factory=list)

    # Phase 6 — Chief synthesis output
    report: str = ""

    # Pipeline metadata
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Dict-like interface for LangGraph compatibility
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# Singleton agent cache to keep inference fast
_seg_agent = None
_clf_agent = None


def get_segmentation_agent():
    global _seg_agent
    if _seg_agent is None:
        from agents.segmentation_agent import SegmentationAgent
        _seg_agent = SegmentationAgent()
    return _seg_agent


def get_classification_agent():
    global _clf_agent
    if _clf_agent is None:
        from agents.classification_agent import ClassificationAgent
        _clf_agent = ClassificationAgent()
    return _clf_agent


def run_pipeline(image_path: str, config_path: str = "configs/config.yaml") -> MedBoardState:
    """
    Run the active MedBoard agents sequentially on an input MRI image.

    Args:
        image_path: Path to the input MRI image (.jpg or .png).
        config_path: Path to config.yaml.

    Returns:
        MedBoardState populated with segmentation and classification outputs.
    """
    state = MedBoardState(image_path=str(image_path))

    if not Path(image_path).exists():
        state.errors.append(f"Image not found at path: {image_path}")
        return state

    # Stage 1: Segmentation Agent (U-Net)
    try:
        seg_agent = get_segmentation_agent()
        state = seg_agent.run(state)
        # Standardize mask as numpy array if tensor returned
        if hasattr(state.tumor_mask, "cpu"):
            mask_arr = state.tumor_mask.cpu().numpy()
            if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
                mask_arr = mask_arr.squeeze(0)
            state.tumor_mask = mask_arr.astype(np.uint8)
    except Exception as e:
        state.errors.append(f"[Segmentation] Error: {str(e)}")

    # Stage 2: Classification Agent (EfficientNet-B0)
    try:
        clf_agent = get_classification_agent()
        state = clf_agent.run(state)
    except Exception as e:
        state.errors.append(f"[Classification] Error: {str(e)}")

    return state


if __name__ == "__main__":
    print("MedBoard pipeline loaded.")
    print("State fields:", list(MedBoardState.__dataclass_fields__.keys()))
