"""
test_preprocessing.py
=====================
Unit tests for data loading, image normalization, and dataset splitting.
"""

import os
import sys
import numpy as np
from PIL import Image

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MAJOR_DIR = os.path.dirname(TEST_DIR)
TRAIN_DIR = os.path.join(MAJOR_DIR, "training")
sys.path.insert(0, TRAIN_DIR)

from preprocessing import load_raw_dataset, create_splits, preprocess_single_image, CATEGORIES


def test_load_raw_dataset():
    data_dir = os.path.join(MAJOR_DIR, "data", "documents")
    X, y, class_to_idx = load_raw_dataset(data_dir)

    assert len(X) == 150
    assert len(y) == 150
    assert X.shape[1:] == (224, 224, 3)
    assert len(class_to_idx) == 5
    assert set(class_to_idx.keys()) == set(CATEGORIES)
    # Input range [0.0, 255.0] float32
    assert X.min() >= 0.0
    assert X.max() <= 255.0


def test_create_splits_mutually_exclusive():
    X = np.zeros((100, 224, 224, 3), dtype=np.float32)
    y = np.array([i % 5 for i in range(100)], dtype=np.int32)

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y, test_size=0.2, val_size=0.2, random_state=42)

    total_len = len(X_train) + len(X_val) + len(X_test)
    assert total_len == 100
    assert len(X_test) == 20
    assert len(X_val) == 20
    assert len(X_train) == 60

    # Verify class balance across all splits
    assert len(np.unique(y_train)) == 5
    assert len(np.unique(y_val)) == 5
    assert len(np.unique(y_test)) == 5


def test_preprocess_single_image():
    img = Image.new("RGB", (300, 500), color=(100, 150, 200))
    tensor = preprocess_single_image(img)

    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 255.0
