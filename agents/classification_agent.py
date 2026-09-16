"""
MedBoard — agents/classification_agent.py
Phase 3: Classification Agent (LangGraph Node)

This agent wraps the EfficientNet-B0 model as a LangGraph node.

What does the agent do?
  - Receives the MedBoardState (shared pipeline state)
  - Loads the trained EfficientNet-B0 classifier (lazily on first invocation)
  - Preprocesses the MRI scan into (3, 224, 224) with ImageNet normalization
  - Runs forward inference to predict the tumor category:
      - glioma
      - meningioma
      - pituitary
      - no_tumor
  - Updates MedBoardState with:
      - predicted_class: str
      - classification_confidence: float (0.0 to 1.0)
      - class_probabilities: dict[str, float]
  - Passes updated state forward to the next agent (Feature Extraction & RAG)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
import yaml

from modules.classifier import build_classifier, predict_class
from modules.preprocessing import MRIPreprocessor
from pipeline import MedBoardState


class ClassificationAgent:
    """
    LangGraph agent node that classifies tumor type from brain MRI.

    Usage in pipeline:
        agent = ClassificationAgent()
        state = agent.run(state)
        print(state.predicted_class, state.classification_confidence)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.clf_cfg   = cfg["classification"]
        self.train_cfg = cfg["training"]
        self.data_cfg  = cfg["data"]

        self.weights_path = self.clf_cfg.get(
            "weights_path", "models/classification/efficientnet_b0_brisc.pth"
        )
        self.num_classes  = self.clf_cfg.get("num_classes", 4)
        self.class_names  = self.data_cfg.get(
            "classes", ["glioma", "meningioma", "no_tumor", "pituitary"]
        )
        self.device_str   = self.train_cfg.get("device", "cuda")

        self.preprocessor = MRIPreprocessor(config_path)
        self._model: Optional[torch.nn.Module] = None

    def _load_model(self) -> None:
        """Load trained classifier weights from disk lazily on first use."""
        weights_path = Path(self.weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Classification model weights not found at '{weights_path}'.\n"
                f"Please train the model first using notebooks/train_classification.ipynb."
            )

        device = torch.device(
            self.device_str if torch.cuda.is_available() and self.device_str == "cuda" else "cpu"
        )

        model = build_classifier(
            num_classes=self.num_classes,
            pretrained=False,  # Load custom trained weights
        )
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self._model = model
        print(f"[ClassificationAgent] Loaded model weights from {weights_path} onto {device}")

    def run(self, state: MedBoardState) -> MedBoardState:
        """
        Execute classification inference on the MRI slice specified in state.image_path.

        Args:
            state: MedBoardState carrying image_path and prior agent outputs.

        Returns:
            Updated MedBoardState with predicted_class, classification_confidence,
            and class_probabilities.
        """
        if not state.image_path:
            state.errors.append("[ClassificationAgent] Missing image_path in MedBoardState.")
            return state

        try:
            if self._model is None:
                self._load_model()

            # Preprocess image for classification: 224x224 RGB, ImageNet normalization
            preprocessed = self.preprocessor.preprocess_image(state.image_path, mode="classification")

            pred_class, confidence, probs = predict_class(
                model=self._model,
                input_tensor=preprocessed.tensor,
                device=self.device_str,
                class_names=self.class_names,
            )

            state.predicted_class = pred_class
            state.classification_confidence = round(confidence, 4)
            state.class_probabilities = {k: round(v, 4) for k, v in probs.items()}

        except Exception as e:
            state.errors.append(f"[ClassificationAgent] Error during classification: {str(e)}")

        return state


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraph Node Function Wrapper
# ─────────────────────────────────────────────────────────────────────────────

_agent_instance: Optional[ClassificationAgent] = None


def classification_node(state: MedBoardState) -> MedBoardState:
    """
    Standard LangGraph node signature function.

    Reuses a singleton instance of ClassificationAgent to avoid reloading
    weights between pipeline steps.
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ClassificationAgent()
    return _agent_instance.run(state)
