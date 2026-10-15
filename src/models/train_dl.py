"""Train the deep models and save val/test probabilities for evaluation.

Usage:
    python -m src.models.train_dl --model cnn
    python -m src.models.train_dl --model densenet                    # stage 1 + stage 2
    python -m src.models.train_dl --model effnetv2 --img-size 384     # one point of the resolution study
    python -m src.models.train_dl --model densenet --quick            # 1+1 epochs on a subset, smoke test
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.dataset import class_weights, labels, load_manifest, make_dataset
from src.evaluation.cost import EpochTimer, model_cost, peak_rss_gb
from src.evaluation.metrics import evaluate, save_probs
from src.models.cnn_scratch import build_cnn
from src.models.densenet_transfer import build_densenet, freeze_base, unfreeze_top
from src.models.efficientnet_transfer import build_efficientnetv2, layers_from_block
from src.models.registry import effnet_key
from src.utils import config as C
from src.utils.runtime import configure_tf


def callbacks(ckpt_name: str, patience: int, extra=()):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(1, patience // 2), min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(str(C.MODELS_DIR / ckpt_name), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(C.RESULTS_DIR / f"{ckpt_name.split('.')[0]}_history.csv")),
        *extra,
    ]


def compile_(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def plot_history(histories: list[tuple[str, dict]], path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    offset = 0
    for stage, h in histories:
        ep = np.arange(offset, offset + len(h["loss"])) + 1
        axes[0].plot(ep, h["loss"], label=f"{stage} train")
        axes[0].plot(ep, h["val_loss"], "--", label=f"{stage} val")
        axes[1].plot(ep, h["accuracy"], label=f"{stage} train")
        axes[1].plot(ep, h["val_accuracy"], "--", label=f"{stage} val")
        offset += len(h["loss"])
    for ax, t in zip(axes, ("Loss", "Accuracy")):
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predict_and_eval(model, key, pretty, val_ds, test_ds, y_test):
    save_probs(key, "val", model.predict(val_ds, verbose=0))
    p_te = model.predict(test_ds, verbose=0)
    save_probs(key, "test", p_te)
    return evaluate(key, y_test, p_te, pretty)


def train_effnetv2(args, img_size, train_ds, val_ds, test_ds, y_test, cw):
    """One point of the resolution study: two-stage EfficientNetV2 training at `img_size`.

    Same recipe as DenseNet121 v2 (stage 1: frozen backbone, lr 1e-3; stage 2: top blocks
    unfrozen with BatchNorm kept frozen, lr --ft-lr), so the only variable across the study
    is the input resolution. Also records time, memory and model cost to <key>_cost.json.
    """
    s1_epochs = 1 if args.quick else (args.epochs or 25)
    s2_epochs = 1 if args.quick else args.ft_epochs
    k1, k2 = effnet_key(img_size, "frozen"), effnet_key(img_size, "finetuned")
    v = C.EFFNET_VARIANT

    model, base_layers = build_efficientnetv2(img_size)
    freeze_base(base_layers)
    compile_(model, 1e-3)
    t1 = EpochTimer()
    start = time.perf_counter()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=s1_epochs, class_weight=cw,
                   callbacks=callbacks(f"{k1}.keras", patience=6, extra=[t1]))
    stage1_s = time.perf_counter() - start
    predict_and_eval(model, k1, f"EfficientNetV2-{v} {img_size}px (frozen)", val_ds, test_ds, y_test)

    n_unfreeze = layers_from_block(base_layers, C.EFFNET_UNFREEZE_FROM)
    unfreeze_top(base_layers, n_unfreeze)
    print(f"Stage 2: {sum(l.trainable for l in base_layers)} base layers trainable (from {C.EFFNET_UNFREEZE_FROM})")
    compile_(model, args.ft_lr)
    t2 = EpochTimer()
    start = time.perf_counter()
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=s2_epochs, class_weight=cw,
                   callbacks=callbacks(f"{k2}.keras", patience=6, extra=[t2]))
    stage2_s = time.perf_counter() - start
    plot_history([("frozen", h1.history), ("fine-tune", h2.history)], C.FIGURES_DIR / f"history_{k2}.png")
    predict_and_eval(model, k2, f"EfficientNetV2-{v} {img_size}px (fine-tuned)", val_ds, test_ds, y_test)

    cost = {
        **model_cost(model, img_size),
        "variant": v, "batch_size": args.batch_size,
        "stage1_epochs": len(t1.epoch_seconds), "stage2_epochs": len(t2.epoch_seconds),
        # first epoch includes decoding + caching the images, so report the median of the rest too
        "stage1_s_per_epoch_median": float(np.median(t1.epoch_seconds[1:] or t1.epoch_seconds)),
        "stage2_s_per_epoch_median": float(np.median(t2.epoch_seconds[1:] or t2.epoch_seconds)),
        "train_minutes_total": (stage1_s + stage2_s) / 60,
        "peak_rss_gb": peak_rss_gb(),
        "unfrozen_from": C.EFFNET_UNFREEZE_FROM,
        "ft_lr": args.ft_lr,
    }
    (C.RESULTS_DIR / f"{k2}_cost.json").write_text(json.dumps(cost, indent=2))
    (C.RESULTS_DIR / f"{k2}_train_config.json").write_text(json.dumps(vars(args), indent=2))
    print("cost:", json.dumps(cost, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "densenet", "effnetv2"], required=True)
    ap.add_argument("--epochs", type=int, default=None, help="cnn epochs / transfer-model stage-1 epochs")
    ap.add_argument("--ft-epochs", type=int, default=25, help="transfer-model stage-2 epochs")
    ap.add_argument("--unfreeze", type=int, default=202, help="densenet layers to unfreeze in stage 2 (202 = conv4_block13 onwards)")
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="transfer-model stage-2 learning rate")
    ap.add_argument("--img-size", type=int, default=None, help="effnetv2 input resolution (default 224)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"default {C.BATCH_SIZE}; effnetv2 uses {C.EFFNET_BATCH_SIZE} at every resolution so they stay comparable")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    args = ap.parse_args()
    if args.batch_size is None:
        args.batch_size = C.EFFNET_BATCH_SIZE if args.model == "effnetv2" else C.BATCH_SIZE
    img_size = (args.img_size or C.IMG_SIZE) if args.model == "effnetv2" else C.IMG_SIZE

    configure_tf()        # Metal graph-optimizer fix: train on the same maths the app runs
    tf.keras.utils.set_random_seed(C.SEED)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    df = load_manifest()
    if args.quick:
        df = df.groupby(["split", "label"]).head(24)
    train_ds = make_dataset(df, "train", augment=True, batch_size=args.batch_size, img_size=img_size)
    val_ds = make_dataset(df, "val", batch_size=args.batch_size, img_size=img_size)
    test_ds = make_dataset(df, "test", batch_size=args.batch_size, img_size=img_size)
    y_test = labels(df, "test")
    cw = class_weights(df)
    print("class weights:", cw)

    if args.model == "cnn":
        epochs = 1 if args.quick else (args.epochs or 50)
        model = build_cnn()
        compile_(model, 1e-3)
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=cw,
                      callbacks=callbacks("cnn_scratch.keras", patience=7))
        plot_history([("cnn", h.history)], C.FIGURES_DIR / "history_cnn_scratch.png")
        predict_and_eval(model, "cnn_scratch", "CNN (scratch)", val_ds, test_ds, y_test)
        return

    if args.model == "effnetv2":
        train_effnetv2(args, img_size, train_ds, val_ds, test_ds, y_test, cw)
        return

    # --- DenseNet121: stage 1, frozen base, train the head only ---
    s1_epochs = 1 if args.quick else (args.epochs or 25)
    s2_epochs = 1 if args.quick else args.ft_epochs
    model, base_layers = build_densenet()
    freeze_base(base_layers)
    compile_(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=s1_epochs, class_weight=cw,
                   callbacks=callbacks("densenet121_frozen.keras", patience=6))
    predict_and_eval(model, "densenet_frozen", "DenseNet121 (frozen)", val_ds, test_ds, y_test)

    # --- stage 2, unfreeze the top of the base with a tiny LR ---
    unfreeze_top(base_layers, args.unfreeze)
    n_trainable = sum(l.trainable for l in base_layers)
    print(f"Stage 2: {n_trainable} base layers trainable")
    compile_(model, args.ft_lr)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=s2_epochs, class_weight=cw,
                   callbacks=callbacks("densenet121_finetuned.keras", patience=6))
    plot_history([("frozen", h1.history), ("fine-tune", h2.history)], C.FIGURES_DIR / "history_densenet.png")
    predict_and_eval(model, "densenet_finetuned", "DenseNet121 (fine-tuned)", val_ds, test_ds, y_test)
    (C.RESULTS_DIR / "densenet_train_config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
