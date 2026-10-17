"""Week 2: Convolutional Neural Networks and Plant Leaf Disease Detection.

This module provides reusable utilities for:
1. CIFAR-10 data loading with reliable local cache fallback.
2. Plant disease dataset utilities and class mapping.
3. ResNet50 model building, loading, and caching for inference.
4. Input image decoding, validation, and model-specific preprocessing.
5. Inference execution returning predicted class and top-K confidence scores.
"""

import io
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Global model cache to avoid reloading model on every request
_MODEL_CACHE: Optional[Any] = None
_CLASS_NAMES_CACHE: Optional[List[str]] = None

# Default path constants relative to Week-2 root
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "plant_disease_resnet50.keras")
DEFAULT_CLASS_MAPPING_PATH = os.path.join(DEFAULT_MODEL_DIR, "class_indices.json")

# Standard 5 plant disease classes
STANDARD_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]

HUMAN_READABLE_NAMES = {
    "Tomato___Bacterial_spot": "Tomato - Bacterial Spot",
    "Tomato___Early_blight": "Tomato - Early Blight",
    "Tomato___Late_blight": "Tomato - Late Blight",
    "Tomato___Leaf_Mold": "Tomato - Leaf Mold",
    "Tomato___healthy": "Tomato - Healthy Leaf",
}


def load_cifar10_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Loads CIFAR-10 dataset reliably.
    
    Checks local Keras cache first to avoid slow external downloads.
    
    Returns:
        ((x_train, y_train), (x_test, y_test))
        x_train: (50000, 32, 32, 3), uint8
        y_train: (50000, 1), int
        x_test: (10000, 32, 32, 3), uint8
        y_test: (10000, 1), int
    """
    keras_dir = os.path.expanduser("~/.keras/datasets")
    cifar_dir = os.path.join(keras_dir, "cifar-10-batches-py")

    if os.path.exists(os.path.join(cifar_dir, "test_batch")):
        def _load_batch(fpath: str) -> Tuple[np.ndarray, np.ndarray]:
            with open(fpath, "rb") as f:
                d = pickle.load(f, encoding="bytes")
                data = d[b"data"]
                labels = d[b"labels"]
                data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
                return data, np.array(labels, dtype=np.int64)

        x_train_list, y_train_list = [], []
        for i in range(1, 6):
            fpath = os.path.join(cifar_dir, f"data_batch_{i}")
            x, y = _load_batch(fpath)
            x_train_list.append(x)
            y_train_list.append(y)
        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0).reshape(-1, 1)

        fpath = os.path.join(cifar_dir, "test_batch")
        x_test, y_test = _load_batch(fpath)
        y_test = y_test.reshape(-1, 1)
        return (x_train, y_train), (x_test, y_test)
    else:
        from tensorflow.keras.datasets import cifar10
        return cifar10.load_data()


def save_class_mapping(class_to_idx: Dict[str, int], filepath: str = DEFAULT_CLASS_MAPPING_PATH) -> None:
    """Saves class-to-index mapping to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)


def load_class_mapping(filepath: str = DEFAULT_CLASS_MAPPING_PATH) -> List[str]:
    """Loads class mapping and returns list of class names sorted by index."""
    global _CLASS_NAMES_CACHE
    if _CLASS_NAMES_CACHE is not None and filepath == DEFAULT_CLASS_MAPPING_PATH:
        return _CLASS_NAMES_CACHE

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            # Sort class names by integer index
            sorted_classes = [c for c, _ in sorted(mapping.items(), key=lambda item: item[1])]
            if filepath == DEFAULT_CLASS_MAPPING_PATH:
                _CLASS_NAMES_CACHE = sorted_classes
            return sorted_classes

    # Fallback to standard 5 classes
    return STANDARD_CLASSES


def build_resnet50_model(num_classes: int = 5, freeze_backbone: bool = True):
    """Builds ResNet50 transfer learning model.
    
    Uses pretrained ImageNet weights with custom classification head.
    """
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model

    inputs = Input(shape=(224, 224, 3), name="input_image")
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    if freeze_backbone:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dense(128, activation="relu", name="dense_features")(x)
    x = Dropout(0.4, name="head_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ResNet50_PlantDisease")
    return model


def get_plant_disease_model(model_path: Optional[str] = None):
    """Loads and caches the trained plant disease model.
    
    Returns cached model instance if already loaded.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model artifact not found at: '{model_path}'. "
            "Please run notebook 05_plant_disease_detector.ipynb or training script first."
        )

    from tensorflow.keras.models import load_model
    _MODEL_CACHE = load_model(model_path)
    return _MODEL_CACHE


def set_cached_model(model: Any) -> None:
    """Manually sets the cached model (useful for unit testing with mock models)."""
    global _MODEL_CACHE
    _MODEL_CACHE = model


def reset_model_cache() -> None:
    """Clears the cached model and class names."""
    global _MODEL_CACHE, _CLASS_NAMES_CACHE
    _MODEL_CACHE = None
    _CLASS_NAMES_CACHE = None


def validate_and_decode_image(file_bytes: bytes, max_size_bytes: int = 10 * 1024 * 1024) -> Image.Image:
    """Validates image size and decodes bytes into a PIL Image.
    
    Args:
        file_bytes: Raw bytes from uploaded image.
        max_size_bytes: Maximum allowed payload in bytes (default 10MB).
        
    Returns:
        PIL.Image in RGB mode.
        
    Raises:
        ValueError: If file exceeds max size or cannot be decoded as an image.
    """
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    if len(file_bytes) > max_size_bytes:
        raise ValueError(f"Uploaded file exceeds maximum size limit of {max_size_bytes / (1024*1024):.1f} MB.")

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # Verify integrity
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file: {str(e)}")

    # Re-open after verify() (Pillow requirement) and convert to RGB
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img


def preprocess_image_for_resnet(img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocesses a PIL Image for ResNet50 inference.
    
    1. Resizes image to target_size (224, 224) using high-quality bilinear sampling.
    2. Converts to NumPy array with shape (1, 224, 224, 3) and dtype float32.
    3. Applies keras.applications.resnet50.preprocess_input (Caffe-style BGR mean subtraction).
    
    Returns:
        Preprocessed NumPy tensor of shape (1, 224, 224, 3).
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input

    # High quality resize
    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)

    # Ensure shape is (224, 224, 3)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    # Add batch dimension -> (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)

    # Apply official ResNet50 preprocessing
    preprocessed = preprocess_input(batch_array)
    return preprocessed


def predict_image(
    file_bytes: bytes,
    model: Optional[Any] = None,
    class_mapping_path: str = DEFAULT_CLASS_MAPPING_PATH,
    top_k: int = 3
) -> Dict[str, Any]:
    """Full inference pipeline on uploaded raw image bytes.
    
    Args:
        file_bytes: Raw image bytes.
        model: Optional preloaded model. If None, retrieves cached model.
        class_mapping_path: Path to class_indices.json.
        top_k: Number of top candidate classes to return.
        
    Returns:
        Dictionary containing:
        - predicted_class: string identifier
        - predicted_label: human-readable name
        - confidence: float score in [0.0, 1.0]
        - top_predictions: list of dicts with {class, label, score}
    """
    img = validate_and_decode_image(file_bytes)
    input_tensor = preprocess_image_for_resnet(img)

    if model is None:
        model = get_plant_disease_model()

    predictions = model.predict(input_tensor, verbose=0)
    probabilities = np.squeeze(predictions)

    # Ensure probabilities are 1D
    if probabilities.ndim == 0:
        probabilities = np.array([float(probabilities)])

    classes = load_class_mapping(class_mapping_path)
    if len(probabilities) != len(classes):
        raise ValueError(
            f"Model output dimension ({len(probabilities)}) does not match "
            f"number of classes in mapping ({len(classes)})."
        )

    # Sort descending by probability
    ranked_indices = np.argsort(probabilities)[::-1]
    top_indices = ranked_indices[:top_k]

    top_predictions = []
    for idx in top_indices:
        cls_name = classes[idx]
        score = float(probabilities[idx])
        top_predictions.append({
            "class": cls_name,
            "label": HUMAN_READABLE_NAMES.get(cls_name, cls_name.replace("___", " - ").replace("_", " ")),
            "score": score,
            "percentage": f"{score * 100:.2f}%"
        })

    best_idx = ranked_indices[0]
    best_class = classes[best_idx]
    best_score = float(probabilities[best_idx])

    return {
        "predicted_class": best_class,
        "predicted_label": HUMAN_READABLE_NAMES.get(best_class, best_class.replace("___", " - ").replace("_", " ")),
        "confidence": best_score,
        "confidence_percentage": f"{best_score * 100:.2f}%",
        "top_predictions": top_predictions
    }
