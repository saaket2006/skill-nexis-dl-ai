"""
Tests for document classifier model loading, inference, and predictions.
"""

import os
import json
import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "document_classifier.keras")
CLASS_MAP_PATH = os.path.join(PROJECT_ROOT, "models", "class_indices.json")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "models", "test_data.npy")


def test_model_artifact_exists():
    assert os.path.isfile(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    assert os.path.isfile(CLASS_MAP_PATH), f"Class indices not found at {CLASS_MAP_PATH}"
    assert os.path.isfile(METADATA_PATH), f"Metadata not found at {METADATA_PATH}"


def test_model_loading_and_architecture():
    model = keras.models.load_model(MODEL_PATH)
    assert model is not None
    # Model takes RGB (None, 224, 224, 3) and outputs (None, 5)
    input_shape = model.input_shape
    output_shape = model.output_shape
    assert input_shape == (None, 224, 224, 3), f"Unexpected input shape: {input_shape}"
    assert output_shape == (None, 5), f"Unexpected output shape: {output_shape}"


def test_model_prediction_shape_and_sum():
    model = keras.models.load_model(MODEL_PATH)
    dummy_input = np.random.rand(2, 224, 224, 3).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    
    assert preds.shape == (2, 5)
    # Check probabilities sum to approx 1.0 (softmax)
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(2), rtol=1e-5, atol=1e-5)
    # Check probabilities in [0, 1]
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_class_mapping_consistency():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    
    expected_classes = ["form", "invoice", "letter", "receipt", "report"]
    assert sorted(list(class_to_idx.keys())) == expected_classes
    assert set(class_to_idx.values()) == {0, 1, 2, 3, 4}


def test_model_accuracy_on_test_data():
    if not os.path.isfile(TEST_DATA_PATH):
        pytest.skip("Test data file test_data.npy not generated yet.")
        
    data = np.load(TEST_DATA_PATH, allow_pickle=True).item()
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    model = keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Model was verified to achieve high test accuracy on held-out test split
    assert acc >= 0.80, f"Expected test accuracy >= 0.80, got {acc:.4f}"
