"""Resolution study (A): does EfficientNetV2 see more at higher input resolution?

Hypothesis: viral pneumonia shows as fine, diffuse interstitial patterns that
get averaged away at 224px, so higher resolution should help the Viral class
most, while Normal vs Pneumonia (large, obvious opacities) is already easy at
224px. Every run uses the same recipe, seed, split and batch size; only the
input size changes.

Reads, for every resolution r in config.EFFNET_RESOLUTIONS:
    reports/results/effnetv2_<v>_<r>_finetuned_{val,test}_probs.npy   (from train_dl.py)
    reports/results/effnetv2_<v>_<r>_finetuned_cost.json              (from train_dl.py)
Writes:
    reports/resolution_study.md, reports/results/resolution_study.json
    reports/figures/resolution_*.png

Usage: python -m src.evaluation.resolution_study
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation.compare import binary_and_subtype, mcnemar
from src.evaluation.metrics import SHORT, compute_metrics, load_probs, plot_confusion
from src.models.registry import effnet_key
from src.utils import config as C


def collect() -> tuple[list[dict], dict]:
    df = pd.read_csv(C.MANIFEST_PATH)
    y_val = df.loc[df["split"] == "val", "label_idx"].values
    y_te = df.loc[df["split"] == "test", "label_idx"].values
    rows, test_probs = [], {}
    for res in C.EFFNET_RESOLUTIONS:
        key = effnet_key(res, "finetuned")
        if not (C.RESULTS_DIR / f"{key}_test_probs.npy").exists():
            print(f"[skip] {res}px not trained yet")
            continue
        p_val, p_te = load_probs(key, "val"), load_probs(key, "test")
        test_probs[res] = p_te
        m = compute_metrics(y_te, p_te)
        cost_file = C.RESULTS_DIR / f"{key}_cost.json"
        cost = json.loads(cost_file.read_text()) if cost_file.exists() else {}
        rows.append({
            "res": res, "key": key,
            "val_macro_f1": float(f1_score(y_val, p_val.argmax(1), average="macro")),
            "accuracy": m["accuracy"], "accuracy_ci": m["accuracy_ci"],
            "macro_f1": m["macro_f1"], "macro_f1_ci": m["macro_f1_ci"],
            "roc_auc_ovr": m["roc_auc_ovr"], "ece": m["ece"],
            "precision": {c: m["per_class"][c]["precision"] for c in C.CLASS_NAMES},
            "recall": {c: m["per_class"][c]["recall"] for c in C.CLASS_NAMES},
            "f1": {c: m["per_class"][c]["f1-score"] for c in C.CLASS_NAMES},
            **binary_and_subtype(y_te, p_te),
            **cost,
        })
        plot_confusion(y_te, p_te, f"EfficientNetV2-{C.EFFNET_VARIANT} {res}px", C.FIGURES_DIR / f"cm_{key}.png")
    return rows, {"y_test": y_te, "probs": test_probs}


def plot_accuracy(rows, path):
    res = [r["res"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        vals = [r[metric] for r in rows]
        lo = [r[metric] - r[f"{metric}_ci"][0] for r in rows]
        hi = [r[f"{metric}_ci"][1] - r[metric] for r in rows]
        ax.errorbar(res, vals, yerr=[lo, hi], marker="o", capsize=4, label=label)
    ax.set_xticks(res)
    ax.set_xlabel("Input resolution (px)")
    ax.set_title("Overall (test, 95% bootstrap CI)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    for c, s in zip(C.CLASS_NAMES, SHORT):
        ax.plot(res, [r["f1"][c] for r in rows], marker="o", label=s)
    ax.set_xticks(res)
    ax.set_xlabel("Input resolution (px)")
    ax.set_title("F1 per class: who benefits from resolution?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost(rows, path):
    res = [r["res"] for r in rows]
    panels = [("gflops_per_image", "GFLOPs / image"),
              ("latency_ms_per_image_batch32", "Inference ms / image (batch 32)"),
              ("stage2_s_per_epoch_median", "Fine-tune seconds / epoch"),
              ("peak_rss_gb", "Peak process memory (GB)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (k, title) in zip(axes, panels):
        vals = [r.get(k) for r in rows]
        if any(v is None for v in vals):
            ax.set_title(f"{title}\n(not measured)")
            continue
        ax.bar([str(x) for x in res], vals, color="#4c78a8")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("px")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(rows, path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for r in rows:
        x = r.get("latency_ms_per_image_batch32")
        if x is None:
            continue
        ax.scatter(x, r["macro_f1"], s=60)
        ax.annotate(f"{r['res']}px", (x, r["macro_f1"]), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Inference ms / image (batch 32)")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Accuracy vs speed")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(v, f="{:.3f}"):
    return "n/a" if v is None else f.format(v)


def main():
    rows, data = collect()
    if not rows:
        raise SystemExit("No EfficientNetV2 runs found. Train with: python -m src.models.train_dl --model effnetv2 --img-size 224")
    plot_accuracy(rows, C.FIGURES_DIR / "resolution_accuracy.png")
    plot_cost(rows, C.FIGURES_DIR / "resolution_cost.png")
    plot_tradeoff(rows, C.FIGURES_DIR / "resolution_tradeoff.png")

    # pairwise significance vs the lowest resolution
    y, P = data["y_test"], data["probs"]
    base = rows[0]["res"]
    for r in rows:
        r["p_vs_lowest_res"] = None if r["res"] == base else mcnemar(y, P[r["res"]], P[base])
    (C.RESULTS_DIR / "resolution_study.json").write_text(json.dumps(rows, indent=2))

    v = C.EFFNET_VARIANT
    L = [f"# Resolution study: EfficientNetV2-{v}", "",
         "Same split, seed, augmentation, two-stage recipe and batch size "
         f"({rows[0].get('batch_size', C.EFFNET_BATCH_SIZE)}) at every resolution; only the input size changes. "
         f"Test set n = {len(y)}.", "",
         "## Accuracy", "",
         "| Resolution | Val macro-F1 | Test accuracy | Test macro-F1 | ROC-AUC | ECE | McNemar p vs " + f"{base}px |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        p = "—" if r["p_vs_lowest_res"] is None else f"{r['p_vs_lowest_res']:.2g}"
        L.append(f"| {r['res']}px | {r['val_macro_f1']:.3f} | {r['accuracy']:.3f} [{r['accuracy_ci'][0]:.3f}–{r['accuracy_ci'][1]:.3f}] | "
                 f"{r['macro_f1']:.3f} [{r['macro_f1_ci'][0]:.3f}–{r['macro_f1_ci'][1]:.3f}] | {r['roc_auc_ovr']:.3f} | {r['ece']:.3f} | {p} |")
    L += ["", "## Per class (test)", "",
          "| Resolution | " + " | ".join(f"{s} P / R / F1" for s in SHORT) + " | Normal vs Pneumonia acc | Bacterial vs Viral acc |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{r['precision'][c]:.2f} / {r['recall'][c]:.2f} / **{r['f1'][c]:.3f}**" for c in C.CLASS_NAMES]
        L.append(f"| {r['res']}px | " + " | ".join(cells) + f" | {r['binary_accuracy']:.3f} | {r['subtype_accuracy']:.3f} |")
    L += ["", "![accuracy](figures/resolution_accuracy.png)", "",
          "## Computational cost (measured on the training machine)", "",
          "| Resolution | Params | GFLOPs / image | Inference ms / image (batch 1) | Inference ms / image (batch 32) | "
          "Stage-1 s / epoch | Stage-2 s / epoch | Epochs (S1 + S2) | Total train time | Peak memory |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['res']}px | {fmt(r.get('params_total'), '{:,}')} | {fmt(r.get('gflops_per_image'), '{:.2f}')} | "
                 f"{fmt(r.get('latency_ms_batch1'), '{:.1f}')} | {fmt(r.get('latency_ms_per_image_batch32'), '{:.2f}')} | "
                 f"{fmt(r.get('stage1_s_per_epoch_median'), '{:.0f}')} | {fmt(r.get('stage2_s_per_epoch_median'), '{:.0f}')} | "
                 f"{r.get('stage1_epochs', '?')} + {r.get('stage2_epochs', '?')} | {fmt(r.get('train_minutes_total'), '{:.0f} min')} | "
                 f"{fmt(r.get('peak_rss_gb'), '{:.1f} GB')} |")
    L += ["", "![cost](figures/resolution_cost.png)", "", "![tradeoff](figures/resolution_tradeoff.png)"]
    out = C.REPORTS_DIR / "resolution_study.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
