"""HOG features for the classical ML baselines (cached to .npz after the first run)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.feature import hog

from src.data.preprocess import load_gray, prepare_gray
from src.utils import config as C


def hog_features(path: str) -> np.ndarray:
    g = prepare_gray(load_gray(path), size=C.ML_IMG_SIZE)
    return hog(g, **C.HOG_PARAMS, feature_vector=True).astype(np.float32)


def load_hog(df: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    cache = C.PROCESSED_DIR / f"hog_{split}_{C.ML_IMG_SIZE}.npz"
    part = df[df["split"] == split]
    if cache.exists():
        d = np.load(cache)
        if len(d["y"]) == len(part):
            return d["X"], d["y"]
    X = np.stack(Parallel(n_jobs=-1)(delayed(hog_features)(p) for p in part["abspath"]))
    y = part["label_idx"].values
    np.savez_compressed(cache, X=X, y=y)
    return X, y
