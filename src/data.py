"""Download and preprocess the published PneumoniaMNIST benchmark splits.

The dataset is a single small NPZ archive. Reading it directly avoids pulling
in MedMNIST's unrelated PyTorch/torchvision dependency stack for this Keras
project.
"""
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "pneumoniamnist.npz"
DATA_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
DATA_MD5 = "28209eda62fecd6e6a2d98b1501bb15f"
IMAGE_SHAPE = (28, 28, 1)
CLASS_NAMES = ("normal", "pneumonia")


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_dataset(download=True):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        if not download:
            raise FileNotFoundError(
                f"Dataset not found at {DATA_FILE}. Run the train command with internet access."
            )
        partial_file = DATA_FILE.with_suffix(".npz.part")
        print("Downloading PneumoniaMNIST from the official Zenodo archive...")
        try:
            urlretrieve(DATA_URL, partial_file)
            actual_md5 = _md5(partial_file)
            if actual_md5 != DATA_MD5:
                raise ValueError(
                    "Downloaded dataset checksum did not match the official file. "
                    "Delete data/pneumoniamnist.npz.part and retry."
                )
            partial_file.replace(DATA_FILE)
        finally:
            if partial_file.exists():
                partial_file.unlink()
    elif _md5(DATA_FILE) != DATA_MD5:
        raise ValueError(
            f"Dataset checksum failed for {DATA_FILE}. Remove that file and run again to redownload it."
        )


def _prepare(images, labels):
    """Return images as float32 [0,1] tensors and labels as 1-D integers."""
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if images.ndim == 3:  # grayscale images have shape (N,H,W)
        images = images[..., np.newaxis]
    if images.ndim != 4:
        raise ValueError(f"Expected a batch of 2-D images; got shape {images.shape}")
    images /= 255.0
    return images, labels


def load_splits(download=True):
    """Download (if needed) and return official train, validation, and test splits."""
    _ensure_dataset(download=download)
    with np.load(DATA_FILE, allow_pickle=False) as archive:
        splits = {
            name: _prepare(archive[f"{name}_images"], archive[f"{name}_labels"])
            for name in ("train", "val", "test")
        }
    return splits
