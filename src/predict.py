"""Run an educational model score on one local image."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.data import CLASS_NAMES
from src.model import build_model


def predict_image(model_path, image_path):
    model_file, image_file = Path(model_path), Path(image_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}. Train the model first.")
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")
    # EXIF transpose avoids silently rotating some phone images incorrectly.
    with Image.open(image_file) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((28, 28))
        array = np.asarray(image, dtype=np.float32)[np.newaxis, ..., np.newaxis] / 255.0
    model = build_model(compile_model=False)
    model.load_weights(model_file)
    score = float(model.predict(array, verbose=0).reshape(-1)[0])
    predicted = CLASS_NAMES[int(score >= 0.5)]
    print(f"Benchmark model score for pneumonia class: {score:.4f}")
    print(f"Thresholded class: {predicted} (threshold=0.50)")
    print("Educational output only. Not a medical diagnosis or clinical recommendation.")
    return score, predicted
