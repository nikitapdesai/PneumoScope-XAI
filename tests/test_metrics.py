import numpy as np

from src.evaluation.metrics import compute_metrics, expected_calibration_error
from src.evaluation.selective import at_threshold, threshold_for_accuracy
from src.models.calibration import apply_temperature, fit_temperature


def _overconfident(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.5
    logits *= 4                                   # sharpen -> over-confident
    p = np.exp(logits - logits.max(1, keepdims=True))
    return y, p / p.sum(1, keepdims=True)


def test_temperature_scaling_reduces_ece_and_keeps_argmax():
    y, p = _overconfident()
    T = fit_temperature(y, p)
    q = apply_temperature(p, T)
    assert T > 1
    assert (q.argmax(1) == p.argmax(1)).all()
    assert expected_calibration_error(y, q) < expected_calibration_error(y, p)


def test_selective_prediction_improves_accuracy():
    y, p = _overconfident()
    thr = threshold_for_accuracy(y, p, 0.95)
    r = at_threshold(y, p, thr)
    assert r["accuracy_answered"] >= 0.95 > r["accuracy_all"]
    assert 0 < r["coverage"] < 1


def test_metrics_perfect_model():
    y = np.array([0, 1, 2] * 20)
    m = compute_metrics(y, np.eye(3)[y], with_ci=False)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1
