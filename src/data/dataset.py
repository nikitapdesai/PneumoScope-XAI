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


def _gray_loader(img_size: int):
    def _load_gray(path, label):
        img = tf.numpy_function(lambda p: prepare_gray(load_gray(p.decode()), size=img_size), [path], tf.uint8)
        img.set_shape((img_size, img_size))
        return img, label
    return _load_gray


def _to_rgb_float(img, label):
    return tf.cast(tf.repeat(img[..., None], 3, axis=-1), tf.float32), label


def make_dataset(df: pd.DataFrame, split: str, augment: bool = False, batch_size: int = C.BATCH_SIZE,
                 shuffle: bool | None = None, img_size: int = C.IMG_SIZE) -> tf.data.Dataset:
    """Batched (image, label) pairs; images are (img_size, img_size, 3) float32 in [0, 255]."""
    part = df[df["split"] == split]
    ds = tf.data.Dataset.from_tensor_slices((part["abspath"].values, part["label_idx"].values.astype("int32")))
    # Decode + CLAHE once and cache as uint8 grayscale: ~300 MB for the whole dataset at 224px,
    # ~1.5 GB at 512px. The 3-channel float copy is made per batch, after the cache.
    ds = ds.map(_gray_loader(img_size), num_parallel_calls=AUTOTUNE).cache()
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
