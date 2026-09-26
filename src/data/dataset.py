"""tf.data pipelines built from manifest.csv."""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data.preprocess import load_gray, prepare_gray
from src.utils import config as C

AUTOTUNE = tf.data.AUTOTUNE


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(C.MANIFEST_PATH)
    df["abspath"] = df["filepath"].map(lambda p: str(C.ROOT / p))
    return df


def build_augmenter() -> tf.keras.Sequential:
    # No horizontal flip: the heart sits on the left, a mirrored chest is not a realistic X-ray.
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03, fill_mode="constant"),   # about +-10 degrees
        tf.keras.layers.RandomZoom(0.10, fill_mode="constant"),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant"),
        tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augment")


def _load_gray(path, label):
    img = tf.numpy_function(lambda p: prepare_gray(load_gray(p.decode())), [path], tf.uint8)
    img.set_shape((C.IMG_SIZE, C.IMG_SIZE))
    return img, label


def _to_rgb_float(img, label):
    return tf.cast(tf.repeat(img[..., None], 3, axis=-1), tf.float32), label


def make_dataset(df: pd.DataFrame, split: str, augment: bool = False,
                 batch_size: int = C.BATCH_SIZE, shuffle: bool | None = None) -> tf.data.Dataset:
    part = df[df["split"] == split]
    ds = tf.data.Dataset.from_tensor_slices((part["abspath"].values, part["label_idx"].values.astype("int32")))
    # Decode + CLAHE once and cache as 224x224 uint8 grayscale (~300 MB for the whole dataset).
    ds = ds.map(_load_gray, num_parallel_calls=AUTOTUNE).cache()
    if shuffle if shuffle is not None else split == "train":
        ds = ds.shuffle(len(part), seed=C.SEED, reshuffle_each_iteration=True)
    ds = ds.map(_to_rgb_float, num_parallel_calls=AUTOTUNE).batch(batch_size)
    if augment:
        aug = build_augmenter()
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0, 255), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def class_weights(df: pd.DataFrame) -> dict[int, float]:
    y = df.loc[df["split"] == "train", "label_idx"].values
    w = compute_class_weight("balanced", classes=np.arange(C.NUM_CLASSES), y=y)
    return {i: float(v) for i, v in enumerate(w)}


def labels(df: pd.DataFrame, split: str) -> np.ndarray:
    return df.loc[df["split"] == split, "label_idx"].values
