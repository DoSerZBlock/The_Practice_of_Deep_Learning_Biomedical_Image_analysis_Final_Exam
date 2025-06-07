#!/usr/bin/env python3
"""
Breast Ultrasound Classification — Grid Search
=============================================
This script performs a *cartesian product* grid search over

    MODEL  ×  PREPROCESSING_PIPELINE

and records the best validation accuracy for every combination.

Supported CNN back‑bones
------------------------
* EfficientNetB0  (ImageNet weights)
* ResNet50        (ImageNet weights)
* DenseNet121     (ImageNet weights)

Supported preprocessing pipelines
---------------------------------
* **baseline**      – Resize ➜ model’s ``preprocess_input`` ➜ light data‑augmentation
* **grayscale**     – Convert to gray ➜ tile to 3‑ch ➜ preprocess
* **clahe**         – (RGB→LAB) CLAHE on L‑channel ➜ LAB→RGB ➜ preprocess
* **center_crop**   – Central crop 80 % ➜ Resize ➜ preprocess

Outputs
-------
``outputs/`` will contain

* ``{model}_{pipe}_loss.png``           – Training / validation loss curves
* ``{model}_{pipe}_accuracy.png``       – Accuracy curves
* ``metrics.csv``                       – Pivot table (rows=model, cols=pipeline)

Run
---
```bash
python train_busi_grid.py
```
"""

from __future__ import annotations

import shutil
from pathlib import Path
import csv

import kagglehub
import matplotlib
matplotlib.use('Agg')    # 改成非互動式後端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    ResNet50,
)
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model

# ╭──────────────────────────── Paths & constants ─────────────────────────────╮
DATASET_SLUG = "jarintasnim090/busi-corrected"  # verified working
ARCHIVES_DIR = Path("archives")
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS_TOP = 100  # reduce for grid search; adjust as needed

for d in (ARCHIVES_DIR, DATA_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ╭──────────────────────────── Download dataset ──────────────────────────────╮
print(f"Downloading dataset {DATASET_SLUG} …")
zip_path = Path(kagglehub.dataset_download(DATASET_SLUG))
archived_zip = ARCHIVES_DIR / zip_path.name
if not archived_zip.exists():
    shutil.move(str(zip_path), archived_zip)
    print(f"Archive moved to {archived_zip}")
else:
    print("Archive already present — skipping move")

# The BUSI‑Corrected archive already contains the folder; no extraction API.
DATASET_DIR = archived_zip  # KaggleHub yields extracted dir path already
if (DATASET_DIR / "BUSI_Corrected").exists():
    DATASET_DIR = DATASET_DIR / "BUSI_Corrected"
print("Using DATASET_DIR =", DATASET_DIR)

# ╭──────────────────── Gather filenames & stratified split ───────────────────╮
class_names = sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])
filepaths, labels = [], []
for cls_idx, cls_name in enumerate(class_names):
    for img_path in (DATASET_DIR / cls_name).glob("*.png"):
        if "_mask" in img_path.stem:
            continue
        filepaths.append(str(img_path))
        labels.append(cls_idx)
filepaths = np.array(filepaths)
labels = np.array(labels)
train_idx, val_idx = train_test_split(
    np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
)
train_paths, val_paths = filepaths[train_idx], filepaths[val_idx]
train_labels, val_labels = labels[train_idx], labels[val_idx]

# ╭─────────────────────────── Preprocessing zoo ──────────────────────────────╮


def build_preprocessor(base_pre):
    """Return a dict of named img‑processing functions that output float32 tensor."""

    def baseline(img):
        return base_pre(img)

    def grayscale(img):
        gray = tf.image.rgb_to_grayscale(img)
        gray3 = tf.repeat(gray, repeats=3, axis=-1)
        return base_pre(gray3)

    def center_crop(img):
        img = tf.image.central_crop(img, 0.8)
        img = tf.image.resize(img, IMG_SIZE)
        return base_pre(img)

    def clahe(img):
        def _clahe_np(arr):
            arr = arr.astype(np.uint8)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe_op.apply(l)
            lab = cv2.merge((l, a, b))
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return rgb.astype(np.uint8)

        img = tf.numpy_function(_clahe_np, [img], tf.uint8)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMG_SIZE)
        return base_pre(tf.cast(img, tf.float32))

    return {
        "baseline": baseline,
        "grayscale": grayscale,
        "clahe": clahe,
        "center_crop": center_crop,
    }


# ╭──────────────────────────── Model zoo dict ────────────────────────────────╮
MODELS = {
    "efficientnetb0": (EfficientNetB0, effnet_pre),
    "resnet50": (ResNet50, resnet_pre),
    "densenet121": (DenseNet121, densenet_pre),
}

# class weights
cw = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)
class_weight_dict = dict(enumerate(cw))

# metrics container
metrics: dict[str, dict[str, float]] = {m: {} for m in MODELS}

# augmenter (shared)
augmenter = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.2),
        keras.layers.RandomBrightness(0.1),
    ]
)


def make_tf_dataset(paths, labels, img_proc_fn, training=False):
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img_proc_fn(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(1024)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE
        )
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# ╭──────────────────────────── Grid‑search loop ──────────────────────────────╮
for model_name, (ModelClass, base_pre) in MODELS.items():
    preprocessors = build_preprocessor(base_pre)
    for pipe_name, img_proc_fn in preprocessors.items():
        combo = f"{model_name}_{pipe_name}"
        print(f"\n=== Training {combo} ===")

        train_ds = make_tf_dataset(
            train_paths, train_labels, img_proc_fn, training=True
        )
        val_ds = make_tf_dataset(val_paths, val_labels, img_proc_fn, training=False)

        base = ModelClass(
            include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet"
        )
        base.trainable = False

        inputs = Input(shape=IMG_SIZE + (3,))
        x = base(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_TOP,
            class_weight=class_weight_dict,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.3, patience=3
                ),
            ],
            verbose=0,
        )

        # save curves
        epochs_r = range(1, len(history.history["loss"]) + 1)
        # loss
        fig, ax = plt.subplots()
        ax.plot(epochs_r, history.history["loss"], label="train")
        ax.plot(epochs_r, history.history["val_loss"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss ({combo})")
        ax.legend()
        fig.savefig(OUTPUTS_DIR / f"{combo}_loss.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        # acc
        fig, ax = plt.subplots()
        ax.plot(epochs_r, history.history["accuracy"], label="train")
        ax.plot(epochs_r, history.history["val_accuracy"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy ({combo})")
        ax.legend()
        fig.savefig(OUTPUTS_DIR / f"{combo}_accuracy.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        best_val_acc = float(max(history.history["val_accuracy"]))
        metrics[model_name][pipe_name] = best_val_acc
        print(f"Best val_accuracy = {best_val_acc:.4f}")

# ╭────────────────────────────── Save CSV ────────────────────────────────────╮
metrics_df = pd.DataFrame(metrics).T  # rows=model , columns=pipeline
csv_path = OUTPUTS_DIR / "metrics.csv"
metrics_df.to_csv(csv_path, float_format="%.4f")
print("\nGrid search finished →", csv_path.resolve())
