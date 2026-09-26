"""Load the deployed model once, predict with calibrated confidence, explain with Grad-CAM."""
import json
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from src.data.preprocess import to_model_input
from src.models.calibration import apply_temperature, load_temperature
from src.models.gradcam import border_attention, gradcam, overlay
from src.utils import config as C

MODEL_KEY = "densenet_finetuned"
MODEL_FILE = C.MODELS_DIR / "densenet121_finetuned.keras"


@dataclass
class Prediction:
    probs: np.ndarray          # calibrated, in CLASS_NAMES order
    label: str
    confidence: float
    abstain: bool
    heatmap_overlay: np.ndarray
    model_input: np.ndarray
    border_attention: float


class Predictor:
    def __init__(self):
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"{MODEL_FILE} not found. Train it with: python -m src.models.train_dl --model densenet")
        self.model = tf.keras.models.load_model(MODEL_FILE)
        self.temperature = load_temperature(MODEL_KEY)
        abstain_file = C.MODELS_DIR / f"{MODEL_KEY}_abstain.json"
        self.threshold = (json.loads(abstain_file.read_text())["threshold"]
                          if abstain_file.exists() else C.DEFAULT_ABSTAIN_THRESHOLD)

    def predict(self, image: np.ndarray) -> Prediction:
        x = to_model_input(image)
        cam, raw = gradcam(self.model, x, C.GRADCAM_LAYER)
        probs = apply_temperature(raw[None], self.temperature)[0]
        idx = int(probs.argmax())
        return Prediction(
            probs=probs,
            label=C.CLASS_NAMES[idx],
            confidence=float(probs[idx]),
            abstain=bool(probs[idx] < self.threshold),
            heatmap_overlay=overlay(x, cam),
            model_input=x.astype(np.uint8),
            border_attention=border_attention(cam),
        )
