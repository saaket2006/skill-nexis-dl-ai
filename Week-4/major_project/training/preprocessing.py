"""
preprocessing.py
================
Data pipeline, image normalization, augmentation, and split management
for the Document Understanding Capstone.

Inputs are processed as (224, 224, 3) float32 in [0, 255].
The model architecture contains an embedded Rescaling layer that maps
[0, 255] -> [-1, 1] matching MobileNetV2 ImageNet expectations.
"""

import os
import json
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
CATEGORIES = ["form", "invoice", "letter", "receipt", "report"]


def load_raw_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load all document images from category subfolders.
    Returns:
        X: numpy array of shape (N, 224, 224, 3), float32 in range [0, 255]
        y: numpy array of shape (N,), int32 category indices
        class_to_idx: mapping from category string to integer index
    """
    images: List[np.ndarray] = []
    labels: List[int] = []

    class_to_idx = {cat: idx for idx, cat in enumerate(sorted(CATEGORIES))}

    for cat in sorted(CATEGORIES):
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(cat_dir, fname)
                img = Image.open(fpath).convert("RGB").resize(IMG_SIZE)
                arr = np.asarray(img, dtype=np.float32)  # [0, 255]
                images.append(arr)
                labels.append(class_to_idx[cat])

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, class_to_idx


def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.1667,
    val_size: float = 0.1667,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible stratified train, validation, and held-out test splits."""
    # First split off test set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then split validation set (15%) from remaining training data
    rel_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=rel_val_size, stratify=y_train_val, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_single_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a single PIL Image for model inference.
    Returns:
        Tensor of shape (1, 224, 224, 3), float32 in range [0, 255].
    """
    img_rgb = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img_rgb, dtype=np.float32)  # [0, 255]
    return np.expand_dims(arr, axis=0)
