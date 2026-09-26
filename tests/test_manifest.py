from pathlib import Path

import numpy as np
import pandas as pd

from src.data.make_manifest import grouped_stratified_split, label_from_path, patient_id_from_path


def test_labels_from_filenames():
    assert label_from_path(Path("train/NORMAL/IM-0115-0001.jpeg")) == "NORMAL"
    assert label_from_path(Path("train/PNEUMONIA/person1_bacteria_1.jpeg")) == "PNEUMONIA_BACTERIA"
    assert label_from_path(Path("test/PNEUMONIA/person1_virus_6.jpeg")) == "PNEUMONIA_VIRUS"
    assert label_from_path(Path("train/PNEUMONIA/weird.jpeg")) is None


def test_patient_ids():
    assert patient_id_from_path(Path("person1234_bacteria_3131.jpeg")) == "PERSON1234"
    assert patient_id_from_path(Path("person1234_virus_2.jpeg")) == "PERSON1234"
    assert patient_id_from_path(Path("IM-0115-0001.jpeg")) == "IM-0115"
    assert patient_id_from_path(Path("NORMAL2-IM-1427-0001.jpeg")) == "NORMAL2-IM-1427"
    assert patient_id_from_path(Path("odd.jpeg")).startswith("UNK-")


def test_split_is_patient_disjoint_and_stratified():
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(600):
        label = rng.choice(["NORMAL", "PNEUMONIA_BACTERIA", "PNEUMONIA_VIRUS"], p=[0.27, 0.47, 0.26])
        for _ in range(rng.integers(1, 4)):
            rows.append({"patient_id": f"P{pid}", "label": label})
    df = pd.DataFrame(rows)
    split = grouped_stratified_split(df, 0.15, 0.15, seed=42)

    assert (df.assign(split=split).groupby("patient_id")["split"].nunique() == 1).all()
    frac = split.value_counts(normalize=True)
    assert 0.10 < frac["test"] < 0.20 and 0.10 < frac["val"] < 0.20
    overall = df["label"].value_counts(normalize=True)
    for s in ("train", "val", "test"):
        dist = df[split == s]["label"].value_counts(normalize=True)
        assert (dist - overall).abs().max() < 0.06
