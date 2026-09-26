"""DenseNet121 transfer-learning model.

The DenseNet graph is built on top of our own preprocessing layers with
`input_tensor=`, so its layers live directly in the outer model. That keeps
`model.get_layer("relu")` reachable for Grad-CAM (no nested sub-model).
"""
import os

import certifi
import tensorflow as tf
from tensorflow.keras import layers

from src.utils import config as C

# python.org / Homebrew Pythons on macOS often lack a CA bundle -> ImageNet weight download fails.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_densenet(input_size: int = C.IMG_SIZE, num_classes: int = C.NUM_CLASSES,
                   weights: str | None = "imagenet") -> tuple[tf.keras.Model, list[tf.keras.layers.Layer]]:
    inp = layers.Input((input_size, input_size, 3), name="image")   # float [0, 255]
    # Same maths as densenet.preprocess_input ("torch" mode), but as serialisable layers.
    x = layers.Rescaling(1.0 / 255)(inp)
    x = layers.Normalization(mean=IMAGENET_MEAN, variance=[s ** 2 for s in IMAGENET_STD], name="imagenet_norm")(x)

    base = tf.keras.applications.DenseNet121(include_top=False, weights=weights, input_tensor=x)
    base_layers = [l for l in base.layers if not isinstance(l, layers.InputLayer)]

    feat = layers.GlobalAveragePooling2D(name="gap")(base.output)
    h = layers.Dense(256, activation="relu", name="head_dense")(feat)
    h = layers.Dropout(0.5, name="head_dropout")(h)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(h)
    model = tf.keras.Model(inp, out, name="densenet121")
    return model, base_layers


def freeze_base(base_layers) -> None:
    for l in base_layers:
        l.trainable = False


def unfreeze_top(base_layers, n: int) -> None:
    """Unfreeze the last n base layers, keeping every BatchNorm frozen.

    Frozen BN in Keras 3 runs in inference mode, so the ImageNet running stats
    aren't overwritten by our small batches during fine-tuning.
    """
    for l in base_layers:
        l.trainable = False
    for l in base_layers[-n:]:
        if not isinstance(l, layers.BatchNormalization):
            l.trainable = True
