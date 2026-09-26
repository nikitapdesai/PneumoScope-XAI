"""Selective prediction: the model may say "not sure, refer to a radiologist".

For each confidence threshold we record coverage (share of cases the model
answers) and accuracy on those cases. A clinical-style operating point is then
chosen on the *validation* set and reported on test.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def risk_coverage(y: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    order = np.argsort(-conf)
    k = np.arange(1, len(y) + 1)
    coverage = k / len(y)
    accuracy = np.cumsum(correct[order]) / k
    return coverage, accuracy, conf[order]


def threshold_for_accuracy(y_val: np.ndarray, probs_val: np.ndarray, target_acc: float) -> float:
    """Lowest confidence threshold whose answered-cases accuracy on val >= target."""
    cov, acc, thr = risk_coverage(y_val, probs_val)
    ok = np.where(acc >= target_acc)[0]
    return float(thr[ok.max()]) if len(ok) else 1.0


def at_threshold(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    keep = probs.max(1) >= threshold
    return {
        "threshold": float(threshold),
        "coverage": float(keep.mean()),
        "accuracy_answered": float((probs[keep].argmax(1) == y[keep]).mean()) if keep.any() else float("nan"),
        "accuracy_all": float((probs.argmax(1) == y).mean()),
    }


def plot_risk_coverage(y, probs_by_name: dict[str, np.ndarray], path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, probs in probs_by_name.items():
        cov, acc, _ = risk_coverage(y, probs)
        ax.plot(cov, acc, label=name)
    ax.set_xlabel("Coverage (share of X-rays the model answers)")
    ax.set_ylabel("Accuracy on answered X-rays")
    ax.set_title("Selective prediction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
