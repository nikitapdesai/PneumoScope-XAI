import numpy as np

from src.evaluation.conformal import (aps_scores, calibrate, conformal_quantile, evaluate_sets,
                                      predict_sets, random_split_check)


def _noisy_probs(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.0                 # a mediocre classifier
    p = np.exp(logits)
    return y, p / p.sum(1, keepdims=True)


def test_quantile_is_kth_smallest():
    s = np.arange(1, 11) / 10                      # 0.1 .. 1.0, n = 10
    # alpha = 0.2 -> k = ceil(11 * 0.8) = 9 -> 9th smallest = 0.9
    assert conformal_quantile(s, 0.2) == 0.9
    # alpha too small for n = 10 -> must include everything
    assert conformal_quantile(s, 0.01) == float("inf")


def test_aps_scores_are_cumulative_in_rank_order():
    p = np.array([[0.2, 0.5, 0.3]])
    np.testing.assert_allclose(aps_scores(p), [[1.0, 0.5, 0.8]])


def test_marginal_coverage_guarantee_holds():
    y, p = _noisy_probs(6000, seed=1)
    for method in ("lac", "aps", "aps_randomized", "lac_classwise"):
        for alpha in (0.05, 0.1, 0.2):
            q = calibrate(p[:3000], y[:3000], alpha, method)
            r = evaluate_sets(predict_sets(p[3000:], q, method), y[3000:])
            assert r["coverage"] >= 1 - alpha - 0.02, (method, alpha, r["coverage"])


def test_classwise_covers_every_class():
    y, p = _noisy_probs(6000, seed=2)
    q = calibrate(p[:3000], y[:3000], 0.1, "lac_classwise")
    r = evaluate_sets(predict_sets(p[3000:], q, "lac_classwise"), y[3000:])
    assert min(r["per_class_coverage"].values()) >= 0.87


def test_smaller_alpha_gives_bigger_sets():
    y, p = _noisy_probs(4000, seed=3)
    sizes = [evaluate_sets(predict_sets(p[2000:], calibrate(p[:2000], y[:2000], a, "lac"), "lac"), y[2000:])["avg_set_size"]
             for a in (0.3, 0.1, 0.02)]
    assert sizes[0] <= sizes[1] <= sizes[2]


def test_random_split_check_reports_coverage_near_target():
    y, p = _noisy_probs(2000, seed=4)
    r = random_split_check(p, y, 0.1, "lac", trials=100)
    assert abs(r["coverage_mean"] - 0.9) < 0.02


def test_aps_sets_are_never_empty():
    # p_max = 0.999 exceeds q_hat = 0.95, which alone would give an empty set
    p = np.array([[0.999, 0.0005, 0.0005]])
    q = np.full(3, 0.95)
    for method in ("aps", "aps_randomized"):
        assert predict_sets(p, q, method)[0, 0]
