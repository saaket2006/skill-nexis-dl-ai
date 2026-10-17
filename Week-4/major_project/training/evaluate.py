"""
evaluate.py
===========
Rigorous evaluation of the trained Document Layout Classifier on the held-out test split.
Generates:
1. Confusion Matrix plot -> reports/confusion_matrix.png
2. One-vs-Rest Multiclass ROC/AUC curve -> reports/roc_curve.png
3. Metrics summary JSON -> reports/evaluation_metrics.json
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def main():
    print("=" * 60)
    print("DOCUMENT UNDERSTANDING CAPSTONE: HELD-OUT TEST EVALUATION")
    print("=" * 60)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load test data
    test_data_path = os.path.join(MODELS_DIR, "test_data.npy")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found at {test_data_path}. Run train.py first.")

    data_dict = np.load(test_data_path, allow_pickle=True).item()
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    print(f"Loaded {len(X_test)} held-out test images.")

    # 2. Load class indices
    with open(os.path.join(MODELS_DIR, "class_indices.json"), "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 3. Load model
    model_path = os.path.join(MODELS_DIR, "document_classifier.keras")
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    # 4. Predict
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"\n================ HELD-OUT TEST RESULTS ================")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print("========================================================")

    # 5. Classification Report
    clf_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Document Layout Classification: Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Label cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.ylabel("True Document Category", fontweight="bold")
    plt.xlabel("Predicted Document Category", fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[OK] Saved Confusion Matrix plot to {cm_path}")

    # 7. One-vs-Rest Multiclass ROC & AUC
    plt.figure(figsize=(8, 6))
    roc_auc_dict = {}
    for i, c_name in enumerate(class_names):
        y_true_binary = (y_test == i).astype(int)
        y_score = y_pred_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        roc_auc_dict[c_name] = round(float(roc_auc), 4)
        plt.plot(fpr, tpr, lw=2, label=f"{c_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title("One-vs-Rest Multiclass ROC Curves", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[OK] Saved ROC Curve plot to {roc_path}")

    # 8. Save Metrics JSON
    metrics_summary = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_acc), 4),
        "macro_avg_f1": round(float(clf_report["macro avg"]["f1-score"]), 4),
        "weighted_avg_f1": round(float(clf_report["weighted avg"]["f1-score"]), 4),
        "roc_auc_one_vs_rest": roc_auc_dict,
        "sample_count": len(X_test),
        "class_names": class_names,
    }
    metrics_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[OK] Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
