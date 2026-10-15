"""Load the deployed model once, predict with calibrated confidence, a conformal prediction set,
and a Grad-CAM explanation.

Which model is served is decided by src/evaluation/compare.py (highest validation macro-F1)
and recorded in models_saved/deployed.json; this file only reads that choice.
"""
import json
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from src.data.preprocess import to_model_input
from src.evaluation.conformal import predict_sets
from src.models.calibration import apply_temperature, load_temperature
from src.models.gradcam import border_attention, gradcam, overlay
from src.models.registry import BY_KEY
from src.utils import config as C

FALLBACK_KEY = "densenet_finetuned"


@dataclass
class Prediction:
    probs: np.ndarray          # temperature-calibrated, in CLASS_NAMES order
    label: str
    confidence: float
    abstain: bool
    prediction_set: list[str]  # conformal set at the requested coverage level
    coverage: float            # 1 - alpha that the set was built for
    heatmap_overlay: np.ndarray
    model_input: np.ndarray
    border_attention: float


class Predictor:
    def __init__(self):
        deployed = C.MODELS_DIR / "deployed.json"
        self.key = json.loads(deployed.read_text())["key"] if deployed.exists() else FALLBACK_KEY
        self.spec = BY_KEY[self.key]
        model_file = C.MODELS_DIR / self.spec.file
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} not found. Train the models, then run: python -m src.evaluation.compare")
        self.model = tf.keras.models.load_model(model_file)
        self.temperature = load_temperature(self.key)
        abstain_file = C.MODELS_DIR / f"{self.key}_abstain.json"
        self.threshold = (json.loads(abstain_file.read_text())["threshold"]
                          if abstain_file.exists() else C.DEFAULT_ABSTAIN_THRESHOLD)
        conformal_file = C.MODELS_DIR / f"{self.key}_conformal.json"
        self.conformal = json.loads(conformal_file.read_text()) if conformal_file.exists() else None

    @property
    def name(self) -> str:
        return self.spec.name

    def conformal_levels(self) -> list[float]:
        """Coverage levels (1 - alpha) the app can offer, from the stored calibration grid."""
        if not self.conformal:
            return []
        grid = self.conformal["q_hat"][self.conformal["default_method"]]
        return sorted(round(1 - float(a), 2) for a in grid)

    def prediction_set(self, raw_probs: np.ndarray, coverage: float) -> list[str]:
        """Conformal set from the RAW softmax (that is what q_hat was calibrated on)."""
        if not self.conformal:
            return []
        method = self.conformal["default_method"]
        q_hat = np.array(self.conformal["q_hat"][method][f"{1 - coverage:.2f}"])
        mask = predict_sets(raw_probs[None], q_hat, method)[0]
        return [c for c, keep in zip(C.CLASS_NAMES, mask) if keep]

    def predict(self, image: np.ndarray, coverage: float | None = None) -> Prediction:
        coverage = coverage or (1 - C.CONFORMAL_ALPHA)
        x = to_model_input(image, self.spec.img_size)
        cam, raw = gradcam(self.model, x, self.spec.gradcam_layer)   # one pass gives heatmap + probs
        probs = apply_temperature(raw[None], self.temperature)[0]
        idx = int(probs.argmax())
        return Prediction(
            probs=probs,
            label=C.CLASS_NAMES[idx],
            confidence=float(probs[idx]),
            abstain=bool(probs[idx] < self.threshold),
            prediction_set=self.prediction_set(raw, coverage),
            coverage=coverage,
            heatmap_overlay=overlay(x, cam),
            model_input=x.astype(np.uint8),
            border_attention=border_attention(cam),
        )
