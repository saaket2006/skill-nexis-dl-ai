"""
train.py
========
Training pipeline for the Document Classification Capstone.
Implements Transfer Learning with an ImageNet-pretrained MobileNetV2 backbone
and a coordinate-preserving spatial classification head.

Preprocessing Convention:
-------------------------
ImageNet-pretrained MobileNetV2 expects pixel values in the range [-1.0, 1.0].
We embed layers.Rescaling(scale=1.0/127.5, offset=-1.0) as the initial layer
so that raw [0, 255] float32 image tensors are scaled identically and deterministically
across training, validation, evaluation, and production REST API inference.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add training folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_raw_dataset, create_splits, CATEGORIES

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "documents")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def build_document_classifier(num_classes: int = 5) -> keras.Model:
    """
    Build Document Layout Classifier using MobileNetV2 with spatial preservation.
    
    1. Input layer accepts (224, 224, 3) float32 in range [0, 255].
    2. Rescaling layer maps [0, 255] -> [-1, 1] matching MobileNetV2 ImageNet convention.
    3. MobileNetV2 frozen feature extractor extracts rich edge and texture maps.
    4. 1x1 Conv + BatchNorm reduces channel depth while preserving 7x7 spatial grid.
    5. Flatten preserves coordinate orientation (top header vs bottom footer).
    6. Dense + Dropout(0.3) + Softmax outputs calibrated class probabilities.
    """
    inputs = layers.Input(shape=(224, 224, 3), name="document_image")

    # Embedded MobileNetV2 preprocessing: [0, 255] -> [-1, 1]
    # Ensures 100% preprocessing consistency between train, eval, and API
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescaling")(inputs)

    # Pretrained MobileNetV2 backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # Feature extraction (output shape: (None, 7, 7, 1280))
    features = base_model(x, training=False)

    # Spatial Feature Preservation Head
    # 1x1 convolution compresses 1280 channels -> 64 channels while retaining 7x7 spatial layout
    x = layers.Conv2D(64, (1, 1), padding="same", activation="relu", name="spatial_conv1x1")(features)
    x = layers.BatchNormalization(name="spatial_bn")(x)

    # Flatten preserves coordinate location (e.g., top-left header vs. bottom barcode)
    x = layers.Flatten(name="spatial_flatten")(x)
    x = layers.Dense(64, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_DocumentLayoutClassifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=" * 65)
    print("DOCUMENT UNDERSTANDING CAPSTONE: MOBILENETV2 MODEL TRAINING")
    print("=" * 65)

    # 1. Load dataset
    print(f"Loading document images from {DATA_DIR}...")
    X, y, class_to_idx = load_raw_dataset(DATA_DIR)
    print(f"Loaded {len(X)} images across {len(class_to_idx)} classes.")

    # 2. Splits (70% train, 15% val, 15% held-out test)
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)
    print(f"Train split: {X_train.shape[0]} | Val split: {X_val.shape[0]} | Held-out Test: {X_test.shape[0]}")

    # Save test set for evaluation script
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(os.path.join(MODELS_DIR, "test_data.npy"), {"X_test": X_test, "y_test": y_test})

    # Save class indices
    with open(os.path.join(MODELS_DIR, "class_indices.json"), "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    # 3. Build model
    model = build_document_classifier(num_classes=len(class_to_idx))
    model.summary()

    # 4. Train with Learning Rate Schedule
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=12,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Evaluate on validation split
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc:.4f} | Validation Loss: {val_loss:.4f}")

    # 6. Save model artifact
    model_save_path = os.path.join(MODELS_DIR, "document_classifier.keras")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # 7. Save metadata
    meta = {
        "model_name": "MobileNetV2_DocumentLayoutClassifier",
        "input_shape": [224, 224, 3],
        "preprocessing": "layers.Rescaling(scale=1.0/127.5, offset=-1.0)",
        "expected_input_range": "[0, 255] float32",
        "mobilenet_internal_range": "[-1.0, 1.0]",
        "categories": list(class_to_idx.keys()),
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "epochs_trained": len(history.history["loss"]),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Model metadata saved successfully!")


if __name__ == "__main__":
    main()
