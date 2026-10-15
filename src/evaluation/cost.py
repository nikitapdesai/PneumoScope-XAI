"""Computational-cost measurements for the resolution study.

Everything here is measured on the machine that ran training (an Apple M5
with the tensorflow-metal GPU plugin in this project), so absolute numbers
are hardware-specific; the *ratios* between resolutions are what matter.
"""
import resource
import sys
import time

import numpy as np
import tensorflow as tf


class EpochTimer(tf.keras.callbacks.Callback):
    """Records wall-clock seconds per epoch (includes the validation pass)."""

    def on_train_begin(self, logs=None):
        self.epoch_seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.perf_counter() - self._t)


def count_flops(model: tf.keras.Model, img_size: int) -> float | None:
    """Forward-pass FLOPs for ONE image (TF profiler convention: 1 multiply-add = 2 FLOPs).

    Freezes the model into a constant graph and lets the v1 profiler count float ops.
    Returns None if the profiler can't handle the graph, rather than guessing.
    """
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        fn = tf.function(lambda x: model(x, training=False))
        concrete = fn.get_concrete_function(tf.TensorSpec([1, img_size, img_size, 3], tf.float32))
        _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(graph_def, name="")
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts["output"] = "none"
            info = tf.compat.v1.profiler.profile(graph=g, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=opts)
        return float(info.total_float_ops)
    except Exception as e:  # noqa: BLE001 - FLOPs are informative only, never fatal
        print(f"[cost] FLOP count failed: {e}")
        return None


def inference_latency(model: tf.keras.Model, img_size: int, batch: int = 1, runs: int = 30, warmup: int = 5) -> float:
    """Median milliseconds per *image* for a forward pass at the given batch size."""
    x = tf.constant(np.random.uniform(0, 255, (batch, img_size, img_size, 3)).astype("float32"))
    for _ in range(warmup):
        model(x, training=False).numpy()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x, training=False).numpy()       # .numpy() forces the GPU work to finish
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000 / batch)


def peak_rss_gb() -> float:
    """Peak resident memory of this process so far (GB). On Apple Silicon, GPU buffers live in the
    same unified memory, so this captures most (not all) of the training footprint."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6   # macOS reports bytes, Linux kilobytes


def model_cost(model: tf.keras.Model, img_size: int) -> dict:
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    flops = count_flops(model, img_size)
    return {
        "img_size": img_size,
        "params_total": int(model.count_params()),
        "params_trainable_final": trainable,
        "gflops_per_image": None if flops is None else flops / 1e9,
        "latency_ms_batch1": inference_latency(model, img_size, batch=1),
        "latency_ms_per_image_batch32": inference_latency(model, img_size, batch=32, runs=10),
    }
