"""Temperature scaling (Guo et al., 2017): one scalar T fit on the validation set.

Softmax outputs are usually over-confident. Dividing logits by T > 1 fixes
that without changing the argmax, so accuracy and F1 stay identical and
only the confidence values become trustworthy.
"""
import json

import numpy as np
from scipy.optimize import minimize_scalar

from src.utils import config as C


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / T
    logits -= logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def fit_temperature(y_val: np.ndarray, probs_val: np.ndarray) -> float:
    def nll(logT):
        p = apply_temperature(probs_val, np.exp(logT))
        return -np.mean(np.log(p[np.arange(len(y_val)), y_val] + 1e-12))
    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(res.x))


def save_temperature(name: str, T: float) -> None:
    (C.MODELS_DIR / f"{name}_temperature.json").write_text(json.dumps({"temperature": T}))


def load_temperature(name: str) -> float:
    p = C.MODELS_DIR / f"{name}_temperature.json"
    return json.loads(p.read_text())["temperature"] if p.exists() else 1.0
