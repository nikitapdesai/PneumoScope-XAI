"""EfficientNetV2 transfer-learning model, built at any input resolution.

Same contract as densenet_transfer.py: the model takes float images in
[0, 255] and does its own normalisation, so preprocess.py and the app need
no model-specific code. Keras' `include_preprocessing=True` adds that
normalisation as layers inside the backbone.

The backbone is built with `input_tensor=`, so its layers live directly in
the outer model and `model.get_layer("top_activation")` stays reachable for
Grad-CAM.
"""
import os

import certifi
import tensorflow as tf
from tensorflow.keras import layers

from src.utils import config as C

# Same macOS CA-bundle fix as densenet_transfer.py, for the ImageNet weight download.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

VARIANTS = {
    "B0": tf.keras.applications.EfficientNetV2B0,
    "S": tf.keras.applications.EfficientNetV2S,
}

# Final activation of the backbone (after top_conv + top_bn): the Grad-CAM target.
GRADCAM_LAYER = "top_activation"


def build_efficientnetv2(img_size: int, variant: str = C.EFFNET_VARIANT, num_classes: int = C.NUM_CLASSES,
                         weights: str | None = "imagenet") -> tuple[tf.keras.Model, list[tf.keras.layers.Layer]]:
    """Returns (model, backbone_layers). Head mirrors the DenseNet one so the two are comparable."""
    inp = layers.Input((img_size, img_size, 3), name="image")   # float [0, 255]
    base = VARIANTS[variant](include_top=False, weights=weights, input_tensor=inp, include_preprocessing=True)
    base_layers = [l for l in base.layers if not isinstance(l, layers.InputLayer)]

    feat = layers.GlobalAveragePooling2D(name="gap")(base.output)
    h = layers.Dense(256, activation="relu", name="head_dense")(feat)
    h = layers.Dropout(0.5, name="head_dropout")(h)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(h)
    model = tf.keras.Model(inp, out, name=f"efficientnetv2_{variant.lower()}_{img_size}")
    return model, base_layers


def layers_from_block(base_layers, block_prefix: str) -> int:
    """How many trailing layers to unfreeze so that `block_prefix` (e.g. "block5a") and everything after train."""
    names = [l.name for l in base_layers]
    first = next(i for i, n in enumerate(names) if n.startswith(block_prefix))
    return len(names) - first
