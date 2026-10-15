"""Grad-CAM figures + a shortcut-learning audit on the test set.

Saves:
  reports/figures/gradcam_grid_<model>.png  (per class: correct and wrong examples)
  reports/gradcam_analysis.md               (border-attention stats)

Usage: python -m src.evaluation.gradcam_report [--model <registry key>] [--max 400]
       (default model = the deployed one, models_saved/deployed.json)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.preprocess import path_to_model_input
from src.evaluation.conformal import deployed_key
from src.evaluation.metrics import SHORT, load_probs
from src.models.gradcam import border_attention, gradcam, overlay
from src.models.registry import BY_KEY
from src.utils import config as C

CAM_MODELS = [k for k, s in BY_KEY.items() if s.gradcam_layer and s.file]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=CAM_MODELS)
    ap.add_argument("--max", type=int, default=400, help="test images to audit (Grad-CAM is one backward pass each)")
    args = ap.parse_args()

    args.model = args.model or deployed_key()
    spec = BY_KEY[args.model]
    layer, size = spec.gradcam_layer, spec.img_size
    model = tf.keras.models.load_model(C.MODELS_DIR / spec.file)

    df = pd.read_csv(C.MANIFEST_PATH)
    test = df[df["split"] == "test"].reset_index(drop=True)
    test["abspath"] = test["filepath"].map(lambda p: str(C.ROOT / p))
    probs = load_probs(args.model, "test")
    test["pred"] = probs.argmax(1)
    test["conf"] = probs.max(1)
    test["correct"] = test["pred"] == test["label_idx"]

    # ---------- figure: for each true class, 2 confident-correct + 2 wrong ----------
    rows = []
    for c in range(C.NUM_CLASSES):
        sub = test[test["label_idx"] == c]
        rows += list(sub[sub["correct"]].nlargest(2, "conf").itertuples())
        rows += list(sub[~sub["correct"]].nlargest(2, "conf").itertuples())
    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)))
    for r, row in enumerate(rows):
        img = path_to_model_input(row.abspath, size)
        cam, p = gradcam(model, img, layer)
        axes[r, 0].imshow(img.astype(np.uint8))
        axes[r, 1].imshow(overlay(img, cam))
        tag = "correct" if row.correct else "WRONG"
        axes[r, 0].set_title(f"true: {SHORT[row.label_idx]}", fontsize=9)
        axes[r, 1].set_title(f"pred: {SHORT[row.pred]} ({p.max():.0%}) {tag}", fontsize=9,
                             color="black" if row.correct else "crimson")
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    grid_path = C.FIGURES_DIR / f"gradcam_grid_{args.model}.png"
    fig.savefig(grid_path, dpi=110)
    plt.close(fig)

    # ---------- audit: where does the model look? ----------
    sample = test.sample(min(args.max, len(test)), random_state=C.SEED)
    sample["border_attention"] = [border_attention(gradcam(model, path_to_model_input(p, size), layer)[0])
                                  for p in sample["abspath"]]
    by_class = sample.groupby("label")["border_attention"].mean()
    by_correct = sample.groupby("correct")["border_attention"].mean()
    # Border strip (12% margin) is ~42% of the image area, so a uniform heatmap would score ~0.42.
    lines = [
        f"# Grad-CAM analysis: {args.model}",
        "",
        f"![grid](figures/{grid_path.name})",
        "",
        "## Shortcut-learning audit",
        "",
        "*Border attention* = share of Grad-CAM mass within 12% of the image edge, where hospital "
        "text markers, arrows and tubes sit. That strip covers ~42% of the image, so a uniform "
        "heatmap would score ~0.42. Lower means the model focuses on the central chest.",
        "",
        f"Audited {len(sample)} test images. Mean border attention: **{sample['border_attention'].mean():.3f}**",
        "",
        "| Group | Mean border attention |",
        "|---|---|",
        *[f"| true {k} | {v:.3f} |" for k, v in by_class.items()],
        *[f"| {'correct' if k else 'misclassified'} | {v:.3f} |" for k, v in by_correct.items()],
        "",
        "If misclassified images show clearly higher border attention, the model is partly using "
        "non-anatomical cues on those cases, which is worth mentioning under limitations.",
    ]
    out = C.REPORTS_DIR / f"gradcam_analysis_{args.model}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
