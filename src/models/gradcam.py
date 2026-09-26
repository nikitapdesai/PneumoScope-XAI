"""Grad-CAM + a simple shortcut-learning check.

`border_attention()` measures how much of the heatmap falls near the image
edges, where hospital markers, text and tubes sit. A model that
really reads the lungs should put most of its attention in the centre.
"""
import cv2
import numpy as np
import tensorflow as tf


_grad_models: dict = {}


def _grad_model(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    key = (id(model), layer_name)
    if key not in _grad_models:
        _grad_models[key] = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    return _grad_models[key]


def gradcam(model: tf.keras.Model, image: np.ndarray, layer_name: str, class_idx: int | None = None):
    """image: (H, W, 3) float in [0, 255]. Returns (heatmap in [0,1] at image size, probs)."""
    grad_model = _grad_model(model, layer_name)
    x = tf.convert_to_tensor(image[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv, probs = grad_model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(probs[0]))
        score = probs[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))                 # (1, C)
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cam.astype(np.float32), probs[0].numpy()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Returns an RGB uint8 image with the jet-coloured heatmap blended on top."""
    base = np.clip(image, 0, 255).astype(np.uint8)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, -1)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)


def border_attention(cam: np.ndarray, margin: float = 0.12) -> float:
    """Fraction of heatmap mass within `margin` of any image edge (0 = all central)."""
    h, w = cam.shape
    mh, mw = int(h * margin), int(w * margin)
    total = cam.sum() + 1e-8
    centre = cam[mh:h - mh, mw:w - mw].sum()
    return float(1 - centre / total)
