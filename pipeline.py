"""
MedBoard — pipeline.py
End-to-end pipeline entry point (stub).

Full implementation in Phase 8 after all agents are built.
This file defines the expected interface that the UI and tests will call.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─── Shared State Schema ──────────────────────────────────────
# This is the LangGraph state object that flows through all agents.
# Each agent reads from and writes to this state dict.

@dataclass
class MedBoardState:
    """Shared state passed between all LangGraph agents."""

    # Input
    image_path: str = ""

    # Phase 1 — Preprocessing output
    preprocessed_image: Optional[np.ndarray] = None   # (C, H, W) float32

    # Phase 2 — Segmentation output
    tumor_mask: Optional[np.ndarray] = None            # (H, W) binary uint8
    has_tumor: bool = False
    segmentation_confidence: float = 0.0

    # Phase 3 — Classification output
    predicted_class: str = ""
    classification_confidence: float = 0.0
    class_probabilities: dict = field(default_factory=dict)

    # Phase 4 — Feature extraction output
    tumor_features: dict = field(default_factory=dict)

    # Phase 5 — Evidence / RAG output
    evidence_snippets: list = field(default_factory=list)

    # Phase 6 — Chief synthesis output
    report: str = ""

    # Pipeline metadata
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def run_pipeline(image_path: str) -> MedBoardState:
    """
    Run the full MedBoard pipeline on a single MRI image.

    Args:
        image_path: Path to the input MRI image (.jpg or .png).

    Returns:
        MedBoardState with all agent outputs populated.

    Note:
        This is a stub. The LangGraph graph will be wired here in Phase 8.
    """
    state = MedBoardState(image_path=image_path)
    raise NotImplementedError(
        "Pipeline not yet implemented. "
        "LangGraph graph will be wired in Phase 8."
    )
    return state


if __name__ == "__main__":
    # Quick smoke test
    print("MedBoard pipeline stub loaded successfully.")
    print("State schema:", MedBoardState.__dataclass_fields__.keys())
