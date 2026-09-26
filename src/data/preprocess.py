"""Single source of truth for turning an X-ray file/array into model input.

Training (via dataset.py), evaluation and the Streamlit app all call these
functions, so the grayscale -> CLAHE -> resize -> RGB chain can't drift
between them. Model-specific normalisation lives *inside* each Keras model.
"""
import threading

import cv2
import numpy as np

from src.utils import config as C

# CLAHE objects aren't thread-safe and tf.data calls us from many threads: one per thread.
_local = threading.local()


def _clahe() -> cv2.CLAHE:
    if not hasattr(_local, "clahe"):
        _local.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _local.clahe


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    if img.shape[-1] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img[..., 0]


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def prepare_gray(img: np.ndarray, size: int = C.IMG_SIZE, clahe: bool = C.USE_CLAHE) -> np.ndarray:
    """Any image array -> (size, size) uint8 grayscale."""
    g = to_gray(img)
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if clahe:
        g = _clahe().apply(g)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def to_model_input(img: np.ndarray, size: int = C.IMG_SIZE) -> np.ndarray:
    """Any image array -> (size, size, 3) float32 in [0, 255]; what every Keras model expects."""
    g = prepare_gray(img, size)
    return np.repeat(g[..., None], 3, axis=-1).astype(np.float32)


def path_to_model_input(path: str, size: int = C.IMG_SIZE) -> np.ndarray:
    return to_model_input(load_gray(path), size)
