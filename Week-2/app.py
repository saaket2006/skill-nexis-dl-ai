"""Flask Application for Week 2 Plant Disease Detection.

Provides a clean web interface and REST API for plant leaf pathology
classification using fine-tuned ResNet50 transfer learning.
"""

import os
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

import model_utils

app = Flask(__name__)

# Configuration
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB maximum upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "plant_village_5_classes")
SAMPLE_TEST_DIR = os.path.join(DATA_DIR, "test")


def allowed_file(filename: str) -> bool:
    """Checks whether filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    """Serves the LeafGuard AI favicon at root /favicon.ico."""
    response = send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon"
    )
    response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.route("/", methods=["GET"])
def index():
    """Renders the main plant disease detector interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Healthcheck endpoint reporting model availability."""
    model_exists = os.path.exists(model_utils.DEFAULT_MODEL_PATH)
    mapping_exists = os.path.exists(model_utils.DEFAULT_CLASS_MAPPING_PATH)
    status = "ready" if (model_exists and mapping_exists) else "uninitialized"

    return jsonify({
        "status": status,
        "model_path": model_utils.DEFAULT_MODEL_PATH,
        "model_available": model_exists,
        "classes_available": mapping_exists,
        "supported_classes": model_utils.load_class_mapping()
    }), (200 if status == "ready" else 503)


@app.route("/sample/<class_name>", methods=["GET"])
def get_sample(class_name: str):
    """Serves a test specimen image for rapid demonstration."""
    safe_name = secure_filename(class_name)
    class_folder = os.path.join(SAMPLE_TEST_DIR, safe_name)

    if not os.path.isdir(class_folder):
        return jsonify({"error": f"Sample class '{safe_name}' not found."}), 404

    images = [f for f in os.listdir(class_folder) if allowed_file(f)]
    if not images:
        return jsonify({"error": f"No sample images available for class '{safe_name}'."}), 404

    sample_path = os.path.join(class_folder, images[0])
    return send_file(sample_path, mimetype="image/jpeg")


@app.route("/predict", methods=["POST"])
def predict():
    """Runs plant leaf disease inference on uploaded image file."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Please upload an image file."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected. Please choose a plant leaf image."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file format. Only JPEG, PNG, and WebP images are supported."
        }), 400

    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Uploaded file is empty (0 bytes)."}), 400

        # Execute prediction pipeline
        result: Dict[str, Any] = model_utils.predict_image(file_bytes)
        return jsonify(result), 200

    except FileNotFoundError as fnf_err:
        return jsonify({
            "error": str(fnf_err),
            "suggestion": "Run notebook 05_plant_disease_detector.ipynb to train and generate the model artifact."
        }), 503

    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400

    except Exception as exc:
        return jsonify({
            "error": f"Inference pipeline failure: {str(exc)}"
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handles upload size exceeding configured limit."""
    return jsonify({
        "error": "File size exceeds 10 MB limit. Please upload a smaller image."
    }), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
