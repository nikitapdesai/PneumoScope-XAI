"""EDA figures.

Usage: python -m src.data.eda
"""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.preprocess import load_gray, prepare_gray
from src.utils import config as C


def main():
    df = pd.read_csv(C.MANIFEST_PATH)
    df["abspath"] = df["filepath"].map(lambda p: str(C.ROOT / p))

    # 1. class distribution per split
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.countplot(data=df, x="label", hue="split", order=C.CLASS_NAMES, ax=ax)
    ax.set_title("Class distribution by split (patient-grouped)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. sample grid: raw vs CLAHE
    rng = np.random.default_rng(C.SEED)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for r, c in enumerate(C.CLASS_NAMES):
        paths = df[df["label"] == c]["abspath"].values[rng.choice((df["label"] == c).sum(), 2, replace=False)]
        for j, p in enumerate(paths):
            raw = load_gray(p)
            axes[r, 2 * j].imshow(cv2.resize(raw, (224, 224)), cmap="gray")
            axes[r, 2 * j + 1].imshow(prepare_gray(raw), cmap="gray")
            axes[r, 2 * j].set_title(f"{c}\nraw", fontsize=8)
            axes[r, 2 * j + 1].set_title("CLAHE", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_samples.png", dpi=120)
    plt.close(fig)

    # 3. image sizes + 4. intensity histograms (on a sample, full reads are slow)
    sub = df.sample(frac=1, random_state=C.SEED).groupby("label").head(300)
    sizes, hists = [], {c: np.zeros(256) for c in C.CLASS_NAMES}
    for p, lab in zip(sub["abspath"], sub["label"]):
        g = load_gray(p)
        sizes.append((g.shape[1], g.shape[0], lab))
        hists[lab] += np.bincount(g.ravel(), minlength=256)
    s = pd.DataFrame(sizes, columns=["width", "height", "label"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.scatterplot(data=s, x="width", y="height", hue="label", s=10, alpha=0.6, ax=axes[0])
    axes[0].set_title("Original resolution")
    for c, h in hists.items():
        axes[1].plot(h / h.sum(), label=c)
    axes[1].set_title("Pixel intensity distribution")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_sizes_intensity.png", dpi=150)
    plt.close(fig)

    print(df.groupby(["label", "split"]).size().unstack())
    print(s.groupby("label")[["width", "height"]].describe().T.round(0))
    print(f"Figures -> {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
