"""Build data/processed/manifest.csv from the raw Kaggle folders.

Relabels the binary dataset into 3 classes using the filenames, recovers a
patient ID for every image, and makes a patient-grouped, class-stratified
train/val/test split so no patient appears in more than one split.

Usage:
    python -m src.data.make_manifest            # expects data/raw/chest_xray/
    python -m src.data.make_manifest --download # fetch from Kaggle first (no API key needed)
"""
import argparse
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils import config as C

IMG_EXTS = {".jpeg", ".jpg", ".png"}

# person1234_bacteria_3131.jpeg -> person1234
_PNEUMONIA_PID = re.compile(r"^(person\d+)_", re.IGNORECASE)
# IM-0115-0001.jpeg -> IM-0115 ; NORMAL2-IM-1427-0001.jpeg -> NORMAL2-IM-1427
_NORMAL_PID = re.compile(r"^((?:NORMAL2-)?IM-\d+)", re.IGNORECASE)


def label_from_path(path: Path) -> str | None:
    folder = path.parent.name.upper()
    name = path.name.lower()
    if folder == "NORMAL":
        return "NORMAL"
    if folder == "PNEUMONIA":
        if "bacteria" in name:
            return "PNEUMONIA_BACTERIA"
        if "virus" in name:
            return "PNEUMONIA_VIRUS"
    return None


def patient_id_from_path(path: Path) -> str:
    # Person numbers are reused across the bacteria/virus and train/test namespaces
    # (person255_bacteria_* and person255_virus_* are visibly different children), but
    # we can't tell which reuses are genuinely the same child. Grouping on the bare
    # number merges them all: coarser groups, but no possible leakage.
    m = _PNEUMONIA_PID.match(path.name) or _NORMAL_PID.match(path.name)
    # Unparseable names become their own group: safe (never merges patients), just less strict.
    return m.group(1).upper() if m else f"UNK-{path.stem}"


def download(raw_dir: Path) -> None:
    """Fetch via kagglehub (public dataset, no API key needed) and symlink into data/raw/."""
    import certifi
    import kagglehub
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    src = Path(kagglehub.dataset_download(C.KAGGLE_DATASET)) / "chest_xray"
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    if not raw_dir.exists():
        raw_dir.symlink_to(src, target_is_directory=True)
    print(f"Dataset at {src} -> {raw_dir}")


def collect(raw_dir: Path) -> pd.DataFrame:
    rows = []
    # Only the top-level train/val/test folders: the Kaggle zip also ships a duplicate
    # chest_xray/chest_xray/ copy and a __MACOSX folder which would double-count images.
    for split in ("train", "val", "test"):
        for p in sorted((raw_dir / split).glob("*/*")):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            label = label_from_path(p)
            if label is None:
                print(f"[skip] cannot label {p}")
                continue
            rows.append({
                "filepath": str(p.relative_to(C.ROOT)),
                "label": label,
                "label_idx": C.CLASS_TO_IDX[label],
                "patient_id": patient_id_from_path(p),
                "original_split": split,
            })
    if not rows:
        raise FileNotFoundError(f"No images found under {raw_dir}. Run with --download or unzip the Kaggle data there.")
    return pd.DataFrame(rows)


def grouped_stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    """70/15/15-style split where every patient lands in exactly one split."""
    split = pd.Series("train", index=df.index)

    k_test = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
    split.iloc[test_idx] = "test"

    rest = df[split == "train"]
    k_val = max(2, round((1 - test_frac) / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(rest, rest["label"], groups=rest["patient_id"]))
    split.loc[rest.index[val_idx]] = "val"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    if args.download:
        download(C.RAW_DIR)

    df = collect(C.RAW_DIR)
    df["split"] = grouped_stratified_split(df, C.VAL_FRAC, C.TEST_FRAC, C.SEED)

    leaked = df.groupby("patient_id")["split"].nunique().gt(1).sum()
    assert leaked == 0, f"{leaked} patients span multiple splits"

    df.to_csv(C.MANIFEST_PATH, index=False)
    print(f"Wrote {len(df)} rows, {df['patient_id'].nunique()} patients -> {C.MANIFEST_PATH}")
    print(pd.crosstab(df["label"], df["split"], margins=True))


if __name__ == "__main__":
    main()
