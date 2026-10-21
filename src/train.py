"""Train and save the baseline model using official train/validation splits."""
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data import load_splits
from src.model import build_model


def train_model(epochs=15, batch_size=32, seed=42):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers")
    tf.keras.utils.set_random_seed(seed)
    splits = load_splits()
    (x_train, y_train) = splits["train"]
    (x_val, y_val) = splits["val"]
    print(f"Train: {x_train.shape}, validation: {x_val.shape}")
    print("Labels: 0=normal, 1=pneumonia")

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}
    model = build_model()
    model.summary()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    model_path = Path("models/pneumonia_cnn.weights.h5")
    temporary_model_path = model_path.with_name("pneumonia_cnn.tmp.weights.h5")
    try:
        model.save_weights(temporary_model_path)
        with open(temporary_model_path, "rb") as saved_file:
            if saved_file.read(8) != b"\x89HDF\r\n\x1a\n":
                raise RuntimeError("Keras did not produce valid HDF5 weights; the saved model was not replaced.")
        os.replace(temporary_model_path, model_path)
    finally:
        if temporary_model_path.exists():
            temporary_model_path.unlink()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, label in (("loss", "Loss"), ("accuracy", "Accuracy")):
        axes[0 if key == "loss" else 1].plot(history.history[key], label="train")
        axes[0 if key == "loss" else 1].plot(history.history[f"val_{key}"], label="validation")
        axes[0 if key == "loss" else 1].set_title(label)
        axes[0 if key == "loss" else 1].set_xlabel("Epoch")
        axes[0 if key == "loss" else 1].legend()
    fig.tight_layout()
    fig.savefig("outputs/training_history.png", dpi=160)
    plt.close(fig)
    print(f"Saved model weights: {model_path}")
    print("Saved plot: outputs/training_history.png")
    return model, history
