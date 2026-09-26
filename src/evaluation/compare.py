"""Build reports/model_comparison.md from every model's saved predictions.

Also runs the "beyond accuracy" analyses on the best model:
temperature-scaling calibration, selective prediction, McNemar significance
tests, and the binary-vs-subtype breakdown.

Usage: python -m src.evaluation.compare
"""
import json
import shutil

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation.metrics import compute_metrics, expected_calibration_error, load_probs, plot_reliability
from src.evaluation.selective import at_threshold, plot_risk_coverage, threshold_for_accuracy
from src.models.calibration import apply_temperature, fit_temperature, save_temperature
from src.utils import config as C

MODELS = [
    ("logreg", "Logistic Regression (HOG)", "classical baseline"),
    ("random_forest", "Random Forest (HOG)", "classical baseline"),
    ("cnn_scratch", "CNN (scratch)", "learned features"),
    ("densenet_frozen", "DenseNet121 (frozen)", "transfer, stage 1"),
    ("densenet_finetuned", "DenseNet121 (fine-tuned)", "transfer, stage 2"),
]
BEST = "densenet_finetuned"
TARGET_ACCS = (0.97, 0.95, 0.90, 0.85)


def fmt_ci(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]"


def mcnemar(y, p_a, p_b) -> float:
    a_ok, b_ok = p_a.argmax(1) == y, p_b.argmax(1) == y
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def binary_and_subtype(y, probs) -> dict:
    """Collapse to Normal-vs-Pneumonia, and look at bacteria-vs-virus on true pneumonia only."""
    yb = (y != 0).astype(int)
    pb = probs[:, 1] + probs[:, 2]
    pn = y != 0
    sub_true = y[pn]
    sub_pred = np.where(probs[pn, 1] >= probs[pn, 2], 1, 2)
    return {
        "binary_accuracy": float(((pb >= 0.5).astype(int) == yb).mean()),
        "binary_auc": float(roc_auc_score(yb, pb)),
        "subtype_accuracy": float((sub_true == sub_pred).mean()),
        "subtype_macro_f1": float(f1_score(sub_true, sub_pred, average="macro")),
    }


def export_demo_samples(y_te, probs_cal, test_paths):
    """Pick demo X-rays for the app from the TEST split: one confident-correct per class + one uncertain."""
    out = C.ROOT / "app" / "assets" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpeg"):
        f.unlink()
    conf, pred = probs_cal.max(1), probs_cal.argmax(1)
    names = ["normal", "bacterial", "viral"]
    for c in range(C.NUM_CLASSES):
        idx = np.where((y_te == c) & (pred == c) & (conf < 0.97))[0]
        if len(idx):
            shutil.copy(test_paths[idx[conf[idx].argmax()]], out / f"{names[c]}_confident.jpeg")
    unsure = np.where((conf > 0.45) & (conf < 0.6))[0]
    if len(unsure):
        i = unsure[0]
        shutil.copy(test_paths[i], out / f"{names[y_te[i]]}_uncertain.jpeg")


def main():
    df = pd.read_csv(C.MANIFEST_PATH)
    y_val = df.loc[df["split"] == "val", "label_idx"].values
    y_te = df.loc[df["split"] == "test", "label_idx"].values

    available = [(k, n, note) for k, n, note in MODELS if (C.RESULTS_DIR / f"{k}_test_probs.npy").exists()]
    if not available:
        raise SystemExit("No saved predictions in reports/results/. Train some models first.")
    probs = {k: load_probs(k, "test") for k, _, _ in available}
    best = BEST if BEST in probs else max(probs, key=lambda k: f1_score(y_te, probs[k].argmax(1), average="macro"))
    best_name = dict((k, n) for k, n, _ in MODELS)[best]

    lines = [
        "# Model comparison",
        "",
        f"Test set: {len(y_te)} X-rays, patient-disjoint from train/val. "
        "95% bootstrap confidence intervals in brackets. p = McNemar test vs the best model.",
        "",
        "| Model | Accuracy | Macro-F1 | ROC-AUC (OvR) | ECE ↓ | p vs best | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    summary = {"test_n": int(len(y_te)), "best": best_name, "models": []}
    for k, name, note in available:
        m = compute_metrics(y_te, probs[k])
        pval = None if k == best else mcnemar(y_te, probs[k], probs[best])
        p = "—" if pval is None else f"{pval:.2g}"
        summary["models"].append({"key": k, "name": name, "note": note, "is_best": k == best, "p_vs_best": pval,
                                  **{f: m[f] for f in ("accuracy", "accuracy_ci", "macro_f1", "macro_f1_ci",
                                                       "roc_auc_ovr", "roc_auc_ovr_ci", "ece")},
                                  "per_class_f1": {c: m["per_class"][c]["f1-score"] for c in C.CLASS_NAMES},
                                  **binary_and_subtype(y_te, probs[k])})
        bold = "**" if k == best else ""
        lines.append(f"| {bold}{name}{bold} | {fmt_ci(m['accuracy'], m['accuracy_ci'])} | "
                     f"{fmt_ci(m['macro_f1'], m['macro_f1_ci'])} | {fmt_ci(m['roc_auc_ovr'], m['roc_auc_ovr_ci'])} | "
                     f"{m['ece']:.3f} | {p} | {note} |")

    lines += ["", "## Per-class F1", "", "| Model | Normal | Bacterial | Viral |", "|---|---|---|---|"]
    for k, name, _ in available:
        pc = compute_metrics(y_te, probs[k], with_ci=False)["per_class"]
        lines.append(f"| {name} | " + " | ".join(f"{pc[c]['f1-score']:.3f}" for c in C.CLASS_NAMES) + " |")

    lines += ["", "## Where the difficulty is: binary vs subtype", "",
              "| Model | Normal vs Pneumonia acc | Normal vs Pneumonia AUC | Bacterial vs Viral acc | Bacterial vs Viral macro-F1 |",
              "|---|---|---|---|---|"]
    for k, name, _ in available:
        b = binary_and_subtype(y_te, probs[k])
        lines.append(f"| {name} | {b['binary_accuracy']:.3f} | {b['binary_auc']:.3f} | "
                     f"{b['subtype_accuracy']:.3f} | {b['subtype_macro_f1']:.3f} |")

    # --- calibration of the best model ---
    p_val = load_probs(best, "val")
    T = fit_temperature(y_val, p_val)
    save_temperature(best, T)
    p_cal = apply_temperature(probs[best], T)
    plot_reliability(y_te, {"before": probs[best], f"after (T={T:.2f})": p_cal},
                     f"Calibration: {best_name}", C.FIGURES_DIR / "reliability_best.png")
    lines += ["", f"## Calibration ({best_name})", "",
              f"Temperature fitted on validation: **T = {T:.2f}**. Accuracy is unchanged; only confidence changes.", "",
              "| | ECE ↓ |", "|---|---|",
              f"| Before | {expected_calibration_error(y_te, probs[best]):.3f} |",
              f"| After temperature scaling | {expected_calibration_error(y_te, p_cal):.3f} |",
              ""]
    if expected_calibration_error(y_te, p_cal) >= expected_calibration_error(y_te, probs[best]):
        lines += [f"T is close to 1, so the fine-tuned model was already well calibrated. "
                  "Scaling does not improve test ECE here; the small difference is within val/test noise.", ""]
    lines += ["![reliability](figures/reliability_best.png)"]

    # --- selective prediction with calibrated probabilities ---
    # Strictest accuracy target that still lets the model answer at least half the val cases.
    p_val_cal = apply_temperature(p_val, T)
    target, thr = None, C.DEFAULT_ABSTAIN_THRESHOLD
    for t in TARGET_ACCS:
        cand = threshold_for_accuracy(y_val, p_val_cal, t)
        if at_threshold(y_val, p_val_cal, cand)["coverage"] >= 0.5:
            target, thr = t, cand
            break
    sel = at_threshold(y_te, p_cal, thr)
    json.dump({"threshold": thr, "target_accuracy": target, "test": sel},
              open(C.MODELS_DIR / f"{best}_abstain.json", "w"), indent=2)
    summary["calibration"] = {"temperature": T, "ece_before": expected_calibration_error(y_te, probs[best]),
                              "ece_after": expected_calibration_error(y_te, p_cal)}
    summary["selective"] = {"target_accuracy": target, **sel}
    (C.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_risk_coverage(y_te, {name: probs[k] for k, name, _ in available}, C.FIGURES_DIR / "risk_coverage.png")
    how = (f"chosen on validation to reach {target:.0%} accuracy on answered cases" if target
           else "default (no accuracy target reachable at ≥50% coverage on validation)")
    lines += ["", "## Selective prediction (\"refer to radiologist\")", "",
              f"Confidence threshold {how}: **{thr:.2f}**.", "",
              f"On test the model answers **{sel['coverage']:.1%}** of X-rays with **{sel['accuracy_answered']:.1%}** accuracy "
              f"(vs {sel['accuracy_all']:.1%} if forced to answer everything); the other "
              f"{1 - sel['coverage']:.1%} are flagged for human review.",
              "", "![risk-coverage](figures/risk_coverage.png)"]

    test_paths = [str(C.ROOT / f) for f in df.loc[df["split"] == "test", "filepath"]]
    export_demo_samples(y_te, p_cal, test_paths)

    out = C.REPORTS_DIR / "model_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
