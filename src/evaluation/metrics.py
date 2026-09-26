"""One evaluation function for every model, so all numbers are apples-to-apples.

Every model saves its val/test probabilities as .npy; `evaluate()` turns those
into metrics (with bootstrap 95% CIs), calibration error and figures.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

from src.utils import config as C

SHORT = ["Normal", "Bacterial", "Viral"]


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_ci(y: np.ndarray, probs: np.ndarray, fn, n: int = 1000, seed: int = C.SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < C.NUM_CLASSES:
            continue
        stats.append(fn(y[idx], probs[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _acc(y, p):
    return accuracy_score(y, p.argmax(1))


def _macro_f1(y, p):
    return f1_score(y, p.argmax(1), average="macro")


def _auc(y, p):
    return roc_auc_score(y, p, multi_class="ovr", average="macro")


def compute_metrics(y: np.ndarray, probs: np.ndarray, with_ci: bool = True) -> dict:
    pred = probs.argmax(1)
    rep = classification_report(y, pred, labels=range(C.NUM_CLASSES), target_names=C.CLASS_NAMES,
                                output_dict=True, zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float(_acc(y, probs)),
        "macro_f1": float(_macro_f1(y, probs)),
        "roc_auc_ovr": float(_auc(y, probs)),
        "ece": expected_calibration_error(y, probs),
        "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score")} for c in C.CLASS_NAMES},
        "confusion_matrix": confusion_matrix(y, pred, labels=range(C.NUM_CLASSES)).tolist(),
    }
    if with_ci:
        out["accuracy_ci"] = bootstrap_ci(y, probs, _acc)
        out["macro_f1_ci"] = bootstrap_ci(y, probs, _macro_f1)
        out["roc_auc_ovr_ci"] = bootstrap_ci(y, probs, _auc)
    return out


def plot_confusion(y, probs, title, path):
    cm = confusion_matrix(y, probs.argmax(1), labels=range(C.NUM_CLASSES), normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=SHORT, yticklabels=SHORT, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y, probs, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for i, name in enumerate(SHORT):
        fpr, tpr, _ = roc_curve(y == i, probs[:, i])
        auc = roc_auc_score(y == i, probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(y, probs_by_name: dict[str, np.ndarray], title, path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    edges = np.linspace(0, 1, n_bins + 1)
    for name, probs in probs_by_name.items():
        conf, correct = probs.max(1), probs.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, "o-", label=f"{name} (ECE {expected_calibration_error(y, probs):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(name: str, y: np.ndarray, probs: np.ndarray, pretty: str | None = None) -> dict:
    """Compute + save metrics JSON and figures for one model's test predictions."""
    pretty = pretty or name
    m = compute_metrics(y, probs)
    m["model"] = pretty
    (C.RESULTS_DIR / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))
    plot_confusion(y, probs, pretty, C.FIGURES_DIR / f"cm_{name}.png")
    plot_roc(y, probs, pretty, C.FIGURES_DIR / f"roc_{name}.png")
    print(f"[{pretty}] acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
          f"AUC={m['roc_auc_ovr']:.4f} ECE={m['ece']:.4f}")
    return m


def save_probs(name: str, split: str, probs: np.ndarray) -> None:
    np.save(C.RESULTS_DIR / f"{name}_{split}_probs.npy", probs.astype(np.float32))


def load_probs(name: str, split: str) -> np.ndarray:
    return np.load(C.RESULTS_DIR / f"{name}_{split}_probs.npy")
