"""TensorFlow runtime settings that must be applied before any graph is built.

tensorflow-metal 1.2.0 (TF 2.18) installs its own Grappler graph optimizer on
Apple GPUs. For EfficientNetV2 and DenseNet the optimized graph returns
numerically different outputs from the same weights on CPU or in eager mode
(see src/evaluation/reference_inference.py and docs/PROJECT_REPORT.md, Section 13).
Training and `model.predict` run as graphs, so without this fix the model is
trained and evaluated on one function and deployed (eager / CPU) as another.

Disabling Grappler's meta-optimizer turns the plugin optimizer off; the GPU then
matches CPU to ~1e-6, at a cost of roughly 15-50% training speed.
"""
import platform
import sys

import tensorflow as tf


def on_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def configure_tf() -> None:
    if on_apple_silicon():
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
