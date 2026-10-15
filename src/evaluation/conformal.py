"""Split conformal prediction: prediction SETS with a coverage guarantee.

Idea
----
Instead of one label, return the set of every class the model can't rule out:
    C(x) = { y : s(x, y) <= q_hat }
where s(x, y) is a *nonconformity score* (how strange label y looks for image x)
and q_hat is a threshold set on a held-out *calibration* set. If calibration and
test images are exchangeable, then for any alpha:

    P( true label in C(x_test) ) >= 1 - alpha            (marginal guarantee)

whatever the model, however badly calibrated it is. A bad model just produces
bigger sets.

Calibration
-----------
1. Score every calibration image with its TRUE label: s_i = s(x_i, y_i), i = 1..n.
2. q_hat = the ceil((n + 1)(1 - alpha)) / n empirical quantile of the s_i
   (the "+1" is the finite-sample correction that makes the guarantee exact).

Scores implemented
------------------
* LAC  (Sadinle et al., 2019):  s(x, y) = 1 - p_y(x)
       Smallest average sets; can return an empty set on very ambiguous inputs.
* APS  (Romano et al., 2020):  s(x, y) = sum of p_j(x) over every class ranked at or above y
       Adapts set size to how spread out the probabilities are, but is conservative.
* Randomized APS: s(x, y) - U * p_y(x), U ~ Uniform(0, 1). Breaks ties at the boundary at
       random so coverage is exact rather than conservative.
  For both APS variants the top-1 label is always kept (as in Angelopoulos et al.'s RAPS code,
  allow_zero_sets=False). Without it, a very confident image whose p_max exceeds q_hat would
  get an EMPTY set. Adding a label can only raise coverage, so the guarantee still holds.
* Class-conditional ("Mondrian") LAC: one q_hat per true class, computed only
       from calibration images of that class, so coverage >= 1 - alpha holds for
       EACH class separately (important for the minority Viral class).

The calibration set here is the validation split (837 X-rays). The model never
trained on it, but it WAS used for early stopping. That is a mild violation of
exchangeability, so an extra check re-splits all held-out data at random many
times to confirm coverage empirically.

Usage: python -m src.evaluation.conformal [--model <registry key>] [--alpha 0.1]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import SHORT, load_probs
from src.models.registry import BY_KEY, SPECS
from src.utils import config as C

METHODS = ("lac", "aps", "aps_randomized", "lac_classwise")
METHOD_NAMES = {"lac": "LAC", "aps": "APS", "aps_randomized": "APS (randomized)", "lac_classwise": "Class-conditional LAC"}
ALPHA_GRID = np.round(np.arange(0.01, 0.31, 0.01), 2)   # stored for the app's confidence slider


# ----------------------------------------------------------------------------- scores
def _as_f64(probs: np.ndarray) -> np.ndarray:
    # Softmax outputs arrive as float32 and don't sum to exactly 1; work in renormalised float64
    # so APS cumulative sums are comparable across images.
    p = probs.astype(np.float64)
    return p / p.sum(1, keepdims=True)


def lac_scores(probs: np.ndarray) -> np.ndarray:
    """(n, K) nonconformity of every candidate label: 1 - p_y."""
    return 1.0 - _as_f64(probs)


def aps_scores(probs: np.ndarray, randomized: bool = False, seed: int = C.SEED) -> np.ndarray:
    """(n, K) APS score of every candidate label: total probability of labels ranked at or above it.

    randomized=True is the version from Romano et al. (2020): subtract a uniform random share
    U * p_y of the label's own mass, so ties at the boundary are broken at random and coverage
    is exact rather than conservative. A fixed seed keeps runs reproducible.
    """
    p = _as_f64(probs)
    order = np.argsort(-p, axis=1)                             # classes, most likely first
    sorted_p = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)                          # mass up to and including rank r
    scores = np.empty_like(p)
    np.put_along_axis(scores, order, cum, axis=1)              # map back to class positions
    if randomized:
        u = np.random.default_rng(seed).uniform(size=(len(p), 1))
        scores = scores - u * p
    return scores


def raps_scores(probs: np.ndarray) -> np.ndarray:
    return aps_scores(probs, randomized=True)


SCORE_FN = {"lac": lac_scores, "aps": aps_scores, "aps_randomized": raps_scores, "lac_classwise": lac_scores}


# ----------------------------------------------------------------------------- calibration
def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n quantile of the true-label calibration scores.

    If that level exceeds 1 (too few calibration points for this alpha), every label must be
    included, so q_hat = +inf.
    """
    n = len(cal_scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))       # rank of the threshold score (1-based)
    if k > n:
        return float("inf")
    return float(np.sort(cal_scores)[k - 1])      # the k-th smallest score, exactly


def calibrate(probs_cal: np.ndarray, y_cal: np.ndarray, alpha: float, method: str) -> np.ndarray:
    """Returns q_hat as a length-K array (identical entries unless method is class-conditional)."""
    s = SCORE_FN[method](probs_cal)[np.arange(len(y_cal)), y_cal]    # score of the TRUE label
    if method == "lac_classwise":
        return np.array([conformal_quantile(s[y_cal == k], alpha) for k in range(probs_cal.shape[1])])
    return np.full(probs_cal.shape[1], conformal_quantile(s, alpha))


def predict_sets(probs: np.ndarray, q_hat: np.ndarray, method: str) -> np.ndarray:
    """(n, K) boolean mask: label k is in the set for image i iff score(i, k) <= q_hat[k]."""
    sets = SCORE_FN[method](probs) <= q_hat[None, :]
    if method.startswith("aps"):
        sets[np.arange(len(probs)), probs.argmax(1)] = True   # never return an empty APS set
    return sets


# ----------------------------------------------------------------------------- evaluation
def evaluate_sets(sets: np.ndarray, y: np.ndarray) -> dict:
    n, K = sets.shape
    covered = sets[np.arange(n), y]
    size = sets.sum(1)
    out = {
        "coverage": float(covered.mean()),
        "avg_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "empty_rate": float((size == 0).mean()),
        "full_set_rate": float((size == K).mean()),
        # accuracy on the images where the set has exactly one label (the "confident" answers)
        "singleton_accuracy": float(covered[size == 1].mean()) if (size == 1).any() else float("nan"),
        "per_class_coverage": {C.CLASS_NAMES[k]: float(covered[y == k].mean()) for k in range(K)},
        "per_class_avg_size": {C.CLASS_NAMES[k]: float(size[y == k].mean()) for k in range(K)},
        "size_distribution": {int(s): float((size == s).mean()) for s in range(K + 1)},
    }
    # which multi-label sets occur, e.g. "Bacterial+Viral"
    combos = pd.Series(["+".join(SHORT[k] for k in range(K) if row[k]) or "∅" for row in sets]).value_counts(normalize=True)
    out["set_composition"] = {k: float(v) for k, v in combos.items()}
    return out


def random_split_check(probs: np.ndarray, y: np.ndarray, alpha: float, method: str,
                       trials: int = 500, seed: int = C.SEED) -> dict:
    """Empirical check of the guarantee: pool all held-out images, split them in half at random
    many times, calibrate on one half and measure coverage on the other."""
    rng = np.random.default_rng(seed)
    cov, size = [], []
    n = len(y)
    for _ in range(trials):
        idx = rng.permutation(n)
        cal, ev = idx[: n // 2], idx[n // 2:]
        q = calibrate(probs[cal], y[cal], alpha, method)
        s = predict_sets(probs[ev], q, method)
        cov.append(s[np.arange(len(ev)), y[ev]].mean())
        size.append(s.sum(1).mean())
    return {"trials": trials, "coverage_mean": float(np.mean(cov)), "coverage_std": float(np.std(cov)),
            "coverage_min": float(np.min(cov)), "frac_trials_below_target": float(np.mean(np.array(cov) < 1 - alpha)),
            "avg_set_size_mean": float(np.mean(size))}


# ----------------------------------------------------------------------------- plots
def plot_alpha_sweep(y_te, p_val, y_val, p_te, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    alphas = ALPHA_GRID[ALPHA_GRID <= 0.3]
    for m in METHODS:
        cov, size = [], []
        for a in alphas:
            s = predict_sets(p_te, calibrate(p_val, y_val, a, m), m)
            r = evaluate_sets(s, y_te)
            cov.append(r["coverage"])
            size.append(r["avg_set_size"])
        axes[0].plot(1 - alphas, cov, "o-", ms=3, label=METHOD_NAMES[m])
        axes[1].plot(1 - alphas, size, "o-", ms=3, label=METHOD_NAMES[m])
    axes[0].plot([0.7, 1], [0.7, 1], "--", color="grey", lw=1, label="target (1 − α)")
    axes[0].set_xlabel("Target coverage 1 − α")
    axes[0].set_ylabel("Empirical test coverage")
    axes[0].set_title("Coverage holds at every level")
    axes[1].set_xlabel("Target coverage 1 − α")
    axes[1].set_ylabel("Average set size (classes)")
    axes[1].set_title("Price of the guarantee: set size")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_set_composition(sets, y, path, title):
    """Stacked bars: for each TRUE class, what kind of set the model returned."""
    K = sets.shape[1]
    labels = ["+".join(SHORT[k] for k in range(K) if row[k]) or "∅" for row in sets]
    df = pd.crosstab(pd.Series([SHORT[i] for i in y], name="true"), pd.Series(labels, name="set"), normalize="index")
    df = df.reindex(SHORT)
    order = sorted(df.columns, key=lambda c: (c.count("+"), c))
    ax = df[order].plot(kind="barh", stacked=True, figsize=(9, 3.4), colormap="tab20", width=0.7)
    ax.set_xlabel("Share of test X-rays")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10)
    ax.legend(title="Prediction set", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------- main
def deployed_key() -> str:
    p = C.MODELS_DIR / "deployed.json"
    return json.loads(p.read_text())["key"] if p.exists() else "densenet_finetuned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="registry key; default = the deployed model (models_saved/deployed.json)")
    ap.add_argument("--alpha", type=float, default=C.CONFORMAL_ALPHA)
    args = ap.parse_args()
    key, alpha = args.model or deployed_key(), args.alpha
    name = BY_KEY[key].name

    df = pd.read_csv(C.MANIFEST_PATH)
    y_val = df.loc[df["split"] == "val", "label_idx"].values
    y_te = df.loc[df["split"] == "test", "label_idx"].values
    # Raw softmax outputs: the guarantee needs no probability calibration.
    p_val, p_te = load_probs(key, "val"), load_probs(key, "test")

    # 1) the three methods at the chosen alpha, calibrated on val, evaluated on test
    results = {}
    for m in METHODS:
        q = calibrate(p_val, y_val, alpha, m)
        sets = predict_sets(p_te, q, m)
        results[m] = {"q_hat": q.tolist(), **evaluate_sets(sets, y_te),
                      "random_split_check": random_split_check(np.vstack([p_val, p_te]),
                                                               np.concatenate([y_val, y_te]), alpha, m)}
        if m == C.CONFORMAL_METHOD:
            plot_set_composition(sets, y_te, C.FIGURES_DIR / "conformal_set_composition.png",
                                 f"{METHOD_NAMES[m]} sets at {1 - alpha:.0%} target coverage: {name}")

    # 2) alpha sweep for this model
    sweep = []
    for a in C.CONFORMAL_ALPHAS:
        for m in METHODS:
            r = evaluate_sets(predict_sets(p_te, calibrate(p_val, y_val, a, m), m), y_te)
            sweep.append({"alpha": a, "method": m, **{k: r[k] for k in
                          ("coverage", "avg_set_size", "singleton_rate", "empty_rate", "singleton_accuracy")},
                          "viral_coverage": r["per_class_coverage"]["PNEUMONIA_VIRUS"]})
    plot_alpha_sweep(y_te, p_val, y_val, p_te, C.FIGURES_DIR / "conformal_alpha_sweep.png",
                     f"Split conformal prediction: {name} (calibrated on val, n={len(y_val)})")

    # 3) every model at the chosen alpha: better models -> smaller sets, same guarantee
    by_model = []
    for spec in SPECS:
        if not (C.RESULTS_DIR / f"{spec.key}_val_probs.npy").exists():
            continue
        pv, pt = load_probs(spec.key, "val"), load_probs(spec.key, "test")
        r = evaluate_sets(predict_sets(pt, calibrate(pv, y_val, alpha, C.CONFORMAL_METHOD), C.CONFORMAL_METHOD), y_te)
        by_model.append({"key": spec.key, "name": spec.name, **{k: r[k] for k in
                         ("coverage", "avg_set_size", "singleton_rate", "singleton_accuracy")},
                         "viral_coverage": r["per_class_coverage"]["PNEUMONIA_VIRUS"]})

    # 4) what the app needs: q_hat for every alpha on the grid, every method
    deploy = {"key": key, "calibration_split": "val", "n_calibration": int(len(y_val)),
              "default_alpha": alpha, "default_method": C.CONFORMAL_METHOD,
              "q_hat": {m: {f"{a:.2f}": calibrate(p_val, y_val, a, m).tolist() for a in ALPHA_GRID} for m in METHODS}}
    (C.MODELS_DIR / f"{key}_conformal.json").write_text(json.dumps(deploy, indent=2))

    summary = {"model": key, "model_name": name, "alpha": alpha, "n_cal": int(len(y_val)), "n_test": int(len(y_te)),
               "methods": results, "sweep": sweep, "by_model": by_model}
    (C.RESULTS_DIR / "conformal_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)


def write_report(S: dict):
    a, R = S["alpha"], S["methods"]
    L = [f"# Conformal prediction: {S['model_name']}", "",
         f"Calibrated on the validation split (n = {S['n_cal']}), evaluated on the test split (n = {S['n_test']}). "
         f"Target coverage 1 − α = **{1 - a:.0%}**.", "",
         "## Methods at the target level", "",
         "| Method | q̂ | Coverage | Avg set size | Singletons | Accuracy on singletons | Empty sets | Viral coverage | Coverage over 500 random re-splits |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        r = R[m]
        q = ", ".join(f"{v:.3f}" for v in r["q_hat"]) if m == "lac_classwise" else f"{r['q_hat'][0]:.3f}"
        rs = r["random_split_check"]
        L.append(f"| {METHOD_NAMES[m]} | {q} | {r['coverage']:.3f} | {r['avg_set_size']:.2f} | {r['singleton_rate']:.1%} | "
                 f"{r['singleton_accuracy']:.1%} | {r['empty_rate']:.1%} | {r['per_class_coverage']['PNEUMONIA_VIRUS']:.3f} | "
                 f"{rs['coverage_mean']:.3f} ± {rs['coverage_std']:.3f} |")
    m = C.CONFORMAL_METHOD
    L += ["", f"## What the sets look like ({METHOD_NAMES[m]})", "",
          "| Set returned | Share of test X-rays |", "|---|---|",
          *[f"| {{{k.replace('+', ', ')}}} | {v:.1%} |" for k, v in R[m]["set_composition"].items()],
          "", "![composition](figures/conformal_set_composition.png)", "",
          "## Coverage vs set size across α", "", "![sweep](figures/conformal_alpha_sweep.png)", "",
          "| α | Method | Coverage | Avg size | Singletons | Viral coverage |", "|---|---|---|---|---|---|",
          *[f"| {r['alpha']:.2f} | {METHOD_NAMES[r['method']]} | {r['coverage']:.3f} | {r['avg_set_size']:.2f} | "
            f"{r['singleton_rate']:.1%} | {r['viral_coverage']:.3f} |" for r in S["sweep"]],
          "", f"## Every model at {1 - a:.0%} target coverage ({METHOD_NAMES[m]})", "",
          "Same guarantee for every model; a better model earns it with smaller sets.", "",
          "| Model | Coverage | Avg set size | Singletons | Accuracy on singletons | Viral coverage |", "|---|---|---|---|---|---|",
          *[f"| {r['name']} | {r['coverage']:.3f} | {r['avg_set_size']:.2f} | {r['singleton_rate']:.1%} | "
            f"{r['singleton_accuracy']:.1%} | {r['viral_coverage']:.3f} |" for r in S["by_model"]]]
    out = C.REPORTS_DIR / "conformal_report.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
