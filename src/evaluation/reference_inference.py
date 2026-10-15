"""Recompute every Keras model's val/test probabilities with REFERENCE maths.

Why this exists
---------------
On Apple GPUs, tensorflow-metal (1.2.0, TF 2.18) runs its own graph optimizer on
every tf.function graph (log line: "Plugin optimizer for device_type GPU is
enabled"). For EfficientNetV2 and DenseNet that optimized graph returns
different numbers than the same weights evaluated on CPU or in eager mode.
Example (EfficientNetV2-B0 384px, one test X-ray, P(Bacterial)):

    CPU (eager, tf.function or predict)    0.801
    Metal GPU, eager                        0.801
    Metal GPU, graph (predict/tf.function)  0.969   <- differs

Disabling Grappler's meta-optimizer (which also switches off the plugin
optimizer) makes the GPU graph match CPU again. Every deployment target
(Streamlit Cloud = Linux CPU, the app's eager call on a Mac) computes the
reference maths, so the reference outputs are the ones to report.

This script saves them as the canonical reports/results/<key>_{val,test}_probs.npy,
after moving the earlier Metal-graph outputs to <key>_{val,test}_probs_metalgraph.npy
so the size of the discrepancy stays documented.

Usage: python -m src.evaluation.reference_inference [--check-cpu 32]
"""
import argparse
import json
import os

import numpy as np

# Must happen before TensorFlow builds any graph. Applied unconditionally here (not only on
# Apple Silicon) because this script's whole point is reference maths.
import tensorflow as tf
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

from src.data.dataset import labels, load_manifest, make_dataset      # noqa: E402
from src.evaluation.metrics import compute_metrics, evaluate          # noqa: E402
from src.models.registry import SPECS                                  # noqa: E402
from src.utils import config as C                                     # noqa: E402


def probs_path(key: str, split: str, suffix: str = "") -> str:
    return str(C.RESULTS_DIR / f"{key}_{split}_probs{suffix}.npy")


def predict(model, df, split, img_size) -> np.ndarray:
    ds = make_dataset(df, split, batch_size=16, img_size=img_size)
    return model.predict(ds, verbose=0)


def cpu_spot_check(model_file: str, df, img_size: int, n: int, gpu_probs: np.ndarray) -> float:
    """Max |GPU-reference - CPU| over the first n test images (run in a CPU-only subprocess-free way)."""
    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(model_file)
        x = np.concatenate([b[0].numpy() for b in make_dataset(df, "test", batch_size=16, img_size=img_size).take(
            (n + 15) // 16)])[:n]
        cpu = model(x, training=False).numpy()
    return float(np.abs(cpu - gpu_probs[:n]).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-cpu", type=int, default=32, help="test images to cross-check against pure CPU (0 = skip)")
    args = ap.parse_args()

    df = load_manifest()
    y_val, y_te = labels(df, "val"), labels(df, "test")
    report = []
    for spec in SPECS:
        if spec.file is None or not spec.file.endswith(".keras"):
            continue
        f = C.MODELS_DIR / spec.file
        if not f.exists():
            print(f"[skip] {spec.key}: {f.name} missing")
            continue
        model = tf.keras.models.load_model(f, compile=False)
        row = {"key": spec.key, "name": spec.name}
        for split, y in (("val", y_val), ("test", y_te)):
            ref = predict(model, df, split, spec.img_size)
            old_path, keep_path = probs_path(spec.key, split), probs_path(spec.key, split, "_metalgraph")
            if os.path.exists(old_path) and not os.path.exists(keep_path):
                os.replace(old_path, keep_path)
            if os.path.exists(keep_path):
                old = np.load(keep_path)
                row[f"{split}_max_abs_diff"] = float(np.abs(old - ref).max())
                row[f"{split}_argmax_agreement"] = float((old.argmax(1) == ref.argmax(1)).mean())
                if split == "test":
                    m_old = compute_metrics(y, old, with_ci=False)
                    row["metalgraph_test_accuracy"], row["metalgraph_test_macro_f1"] = m_old["accuracy"], m_old["macro_f1"]
            np.save(probs_path(spec.key, split), ref.astype(np.float32))
            if split == "test":
                m = evaluate(spec.key, y, ref, spec.name)      # rewrites <key>_metrics.json + cm/roc figures
                row["test_accuracy"], row["test_macro_f1"] = m["accuracy"], m["macro_f1"]
                if args.check_cpu:
                    row["cpu_check_max_abs_diff"] = cpu_spot_check(str(f), df, spec.img_size, args.check_cpu, ref)
        report.append(row)
        print(json.dumps(row))

    (C.RESULTS_DIR / "reference_inference.json").write_text(json.dumps(report, indent=2))
    L = ["# Reference vs Metal-graph inference", "",
         "Test-set metrics of the SAME saved weights, evaluated two ways. *Metal graph* = tensorflow-metal's optimized "
         "graph (what `model.predict` did during training). *Reference* = Grappler meta-optimizer disabled, which matches "
         "CPU (column 'CPU check'). All other reports use the reference numbers.", "",
         "| Model | Metal-graph acc | Reference acc | Metal-graph macro-F1 | Reference macro-F1 | Test argmax agreement | Max prob diff | CPU check (max diff) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in report:
        L.append(f"| {r['name']} | {r.get('metalgraph_test_accuracy', float('nan')):.3f} | {r['test_accuracy']:.3f} | "
                 f"{r.get('metalgraph_test_macro_f1', float('nan')):.3f} | {r['test_macro_f1']:.3f} | "
                 f"{r.get('test_argmax_agreement', float('nan')):.1%} | {r.get('test_max_abs_diff', float('nan')):.3f} | "
                 f"{r.get('cpu_check_max_abs_diff', float('nan')):.1e} |")
    (C.REPORTS_DIR / "reference_inference.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
