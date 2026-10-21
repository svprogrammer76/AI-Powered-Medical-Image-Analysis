"""Evaluate a saved model once on the official test split."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

from src.data import CLASS_NAMES, load_splits
from src.model import build_model


def evaluate_model(model_path="models/pneumonia_cnn.weights.h5"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {path}. Train first with `run_windows.bat train` "
            "on Windows or `.venv/bin/python main.py train` on macOS/Linux."
        )
    (x_test, y_test) = load_splits()["test"]
    model = build_model(compile_model=False)
    model.load_weights(path)
    scores = model.predict(x_test, verbose=0).reshape(-1)
    predictions = (scores >= 0.5).astype(np.int64)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "test_samples": int(len(y_test)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_pneumonia": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_pneumonia": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "confusion_matrix_labels": list(CLASS_NAMES),
        "confusion_matrix_rows_actual_cols_predicted": matrix.tolist(),
        "warning": "Educational benchmark only; not for clinical use.",
    }
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual",
           title="Held-out test confusion matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Display a small random subset for a visual audit; labels are dataset labels.
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(len(y_test), size=min(12, len(y_test)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for ax, index in zip(axes.flat, sample_ids):
        ax.imshow(x_test[index].squeeze(), cmap="gray")
        ax.set_title(f"Label: {CLASS_NAMES[y_test[index]]}\nScore: {scores[index]:.2f}")
        ax.axis("off")
    for ax in axes.flat[len(sample_ids):]:
        ax.axis("off")
    fig.suptitle("Model scores on benchmark images (not diagnoses)")
    fig.tight_layout()
    fig.savefig("outputs/test_predictions.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print("Saved outputs/metrics.json, outputs/confusion_matrix.png, outputs/test_predictions.png")
    return metrics
