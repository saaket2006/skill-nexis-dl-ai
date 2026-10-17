"""
model_utils.py
==============
Reusable inference and artifact loading utilities for Document Understanding.
Coordinates loading trained Keras models, class mapping dictionaries,
and input tensor preprocessing matching the MobileNetV2 ImageNet convention.
"""

import os
import time
import json
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
IMG_SIZE = (224, 224)


def load_model_and_metadata(models_dir: str = MODELS_DIR) -> Tuple[keras.Model, Dict[str, int], Dict[int, str], dict]:
    """Load model, class indices, and metadata from disk."""
    model_path = os.path.join(models_dir, "document_classifier.keras")
    classes_path = os.path.join(models_dir, "class_indices.json")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Class indices file not found at: {classes_path}")

    model = keras.models.load_model(model_path)

    with open(classes_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, class_to_idx, idx_to_class, metadata


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess PIL Image into float32 (1, 224, 224, 3) tensor in range [0, 255].
    The model's internal Rescaling layer converts [0, 255] to [-1, 1] as expected by MobileNetV2.
    """
    rgb_img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(rgb_img, dtype=np.float32)  # [0, 255]
    return np.expand_dims(arr, axis=0)


def predict_document_image(
    model: keras.Model,
    idx_to_class: Dict[int, str],
    image: Image.Image,
    top_k: int = 5,
) -> Tuple[str, float, List[Dict[str, Any]], float]:
    """
    Run forward pass and return predicted category, top confidence,
    ranked top-k predictions, and latency in milliseconds.
    """
    input_tensor = preprocess_image(image)

    t0 = time.perf_counter()
    probs = model.predict(input_tensor, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    top_idx = int(np.argmax(probs))
    top_cat = idx_to_class.get(top_idx, "unknown")
    top_conf = float(probs[top_idx])

    # Rank top-k
    safe_top_k = min(max(top_k, 1), len(idx_to_class))
    ranked_indices = np.argsort(probs)[::-1][:safe_top_k]
    top_k_list = []
    for idx in ranked_indices:
        cat_name = idx_to_class.get(int(idx), f"class_{idx}")
        conf = float(probs[idx])
        top_k_list.append({
            "category": cat_name,
            "confidence": round(conf, 4),
            "percentage": f"{conf * 100.0:.2f}%",
        })

    return top_cat, round(top_conf, 4), top_k_list, round(latency_ms, 2)
