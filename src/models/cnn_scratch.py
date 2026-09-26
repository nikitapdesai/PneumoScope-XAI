"""Custom CNN baseline (4 conv blocks -> GAP -> dense head)."""
import tensorflow as tf
from tensorflow.keras import layers

from src.utils import config as C


def build_cnn(input_size: int = C.IMG_SIZE, num_classes: int = C.NUM_CLASSES) -> tf.keras.Model:
    inp = layers.Input((input_size, input_size, 3), name="image")   # float [0, 255]
    x = layers.Rescaling(1.0 / 255)(inp)
    for i, f in enumerate((32, 64, 128, 256)):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i + 1}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu", name=f"conv{i + 1}_relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inp, out, name="cnn_scratch")


# Grad-CAM target for this model
GRADCAM_LAYER = "conv4_relu"
