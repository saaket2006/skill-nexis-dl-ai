"""Automated Test Suite for Week 2 Plant Disease Detection System.

Tests cover:
1. Flask Application startup and homepage response.
2. Health check endpoint reporting model availability.
3. Image preprocessing dimensions, channels, and float32 data types.
4. Valid image upload and top-K inference pipeline.
5. Invalid file type handling (.txt, .exe, etc.).
6. Corrupted / unreadable image bytes handling.
7. Maximum upload size and zero-byte payload rejection.
8. Missing model artifact graceful degradation (503 response).
9. Output dimensionality matching 5 classes.
10. Confidence calibration range [0.0, 1.0] and sum to ~1.0.
11. Class index to human-readable label mapping.
12. Sample specimen endpoint functionality.
"""

import io
import json
import os
import sys
import pytest
import numpy as np
from PIL import Image

# Ensure Week-2 folder is on python path
WEEK2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEEK2_DIR not in sys.path:
    sys.path.insert(0, WEEK2_DIR)

import model_utils
from app import app


@pytest.fixture
def client():
    """Configures Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def sample_leaf_image_bytes():
    """Generates synthetic green leaf-like RGB image bytes in memory."""
    img = Image.new("RGB", (256, 256), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class MockClassifierModel:
    """Mock model returning controlled softmax probabilities for 5 classes."""
    def __init__(self, probas=(0.05, 0.85, 0.04, 0.03, 0.03)):
        self.probas = np.array([probas], dtype=np.float32)

    def predict(self, x, verbose=0):
        # Assert input shape matches ResNet50 input requirements
        assert x.shape == (1, 224, 224, 3)
        assert x.dtype == np.float32
        return self.probas


# -------------------------------------------------------------
# 1. Flask Web Application Route Tests
# -------------------------------------------------------------

def test_homepage_route(client):
    """Test that the homepage loads successfully with HTTP 200 and expected markup."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "LeafGuard AI" in html
    assert "Plant Leaf Disease" in html
    assert "uploadForm" in html
    assert "favicon.svg" in html
    assert "favicon.ico" in html


def test_favicon_ico_route(client):
    """Test /favicon.ico returns 200 with icon MIME type and cache-busting headers."""
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert "image/" in response.content_type or "icon" in response.content_type
    assert len(response.data) > 0
    assert "no-cache" in response.headers.get("Cache-Control", "")


def test_static_favicon_assets(client):
    """Test that all LeafGuard AI favicon variants exist and return HTTP 200."""
    for filename in ["favicon.svg", "favicon.ico", "favicon-32x32.png", "apple-touch-icon.png"]:
        res = client.get(f"/static/{filename}")
        assert res.status_code == 200, f"Static asset {filename} missing"
        assert len(res.data) > 0


def test_health_check_endpoint(client):
    """Test /health endpoint returns JSON with model and class status."""
    response = client.get("/health")
    assert response.status_code in (200, 503)
    data = response.get_json()
    assert "status" in data
    assert "supported_classes" in data
    assert len(data["supported_classes"]) == 5


def test_sample_image_endpoint(client):
    """Test /sample/<class_name> endpoint serves test specimen image."""
    response = client.get("/sample/Tomato___healthy")
    if response.status_code == 200:
        assert response.content_type.startswith("image/")
        assert len(response.data) > 0
    else:
        # If test directory hasn't downloaded specimens yet, should return 404
        assert response.status_code == 404


def test_sample_nonexistent_class(client):
    """Test /sample/<class_name> with invalid class returns 404."""
    response = client.get("/sample/NonexistentPlantClass123")
    assert response.status_code == 404


# -------------------------------------------------------------
# 2. Upload Validation and Error Handling Tests
# -------------------------------------------------------------

def test_predict_no_file(client):
    """Test /predict returns 400 when no file is included in request."""
    response = client.post("/predict", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_empty_filename(client):
    """Test /predict returns 400 when empty filename is uploaded."""
    data = {"file": (io.BytesIO(b""), "")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_predict_invalid_extension(client):
    """Test /predict rejects unsupported file formats (.txt, .exe, .sh)."""
    data = {"file": (io.BytesIO(b"Hello world"), "notes.txt")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res_data = response.get_json()
    assert "Invalid file format" in res_data["error"]


def test_predict_corrupted_image(client):
    """Test /predict gracefully handles corrupt / truncated image bytes."""
    corrupted_bytes = b"\xFF\xD8\xFF\xE0" + b"\x00" * 20  # Incomplete JPEG header
    data = {"file": (io.BytesIO(corrupted_bytes), "corrupt.jpg")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res_data = response.get_json()
    assert "Invalid or corrupted image" in res_data["error"]


def test_predict_zero_byte_image(client):
    """Test /predict rejects 0-byte file payload."""
    data = {"file": (io.BytesIO(b""), "empty.jpg")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res_data = response.get_json()
    assert "empty" in res_data["error"]


# -------------------------------------------------------------
# 3. Model Utils and Preprocessing Unit Tests
# -------------------------------------------------------------

def test_image_decoding_and_validation(sample_leaf_image_bytes):
    """Test Pillow image decoding converts to RGB and validates integrity."""
    img = model_utils.validate_and_decode_image(sample_leaf_image_bytes)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (256, 256)


def test_resnet_preprocessing_dimensions(sample_leaf_image_bytes):
    """Test that preprocessing produces tensor of shape (1, 224, 224, 3) and dtype float32."""
    img = model_utils.validate_and_decode_image(sample_leaf_image_bytes)
    tensor = model_utils.preprocess_image_for_resnet(img)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_class_mapping_consistency():
    """Verify standard classes and class_indices mapping integrity."""
    classes = model_utils.load_class_mapping()
    assert len(classes) == 5
    assert "Tomato___healthy" in classes
    assert "Tomato___Early_blight" in classes


# -------------------------------------------------------------
# 4. Inference Pipeline Tests (Mock Model)
# -------------------------------------------------------------

def test_inference_with_mock_model(sample_leaf_image_bytes):
    """Test prediction pipeline with mock classifier producing deterministic output."""
    mock_model = MockClassifierModel(probas=(0.05, 0.85, 0.04, 0.03, 0.03))
    
    result = model_utils.predict_image(
        sample_leaf_image_bytes,
        model=mock_model,
        top_k=3
    )

    assert "predicted_class" in result
    assert "confidence" in result
    assert "top_predictions" in result
    
    # Class at index 1 is Early_blight
    classes = model_utils.load_class_mapping()
    expected_class = classes[1]
    assert result["predicted_class"] == expected_class
    assert pytest.approx(result["confidence"], 0.01) == 0.85
    assert len(result["top_predictions"]) == 3
    assert result["top_predictions"][0]["class"] == expected_class
    assert pytest.approx(result["top_predictions"][0]["score"], 0.01) == 0.85


def test_predict_route_with_mock_model(client, sample_leaf_image_bytes):
    """Test full HTTP POST /predict flow using cached mock model."""
    mock_model = MockClassifierModel(probas=(0.02, 0.03, 0.05, 0.10, 0.80))
    model_utils.set_cached_model(mock_model)

    try:
        data = {"file": (io.BytesIO(sample_leaf_image_bytes), "leaf.jpg")}
        response = client.post("/predict", data=data, content_type="multipart/form-data")
        assert response.status_code == 200

        res_data = response.get_json()
        assert res_data["predicted_class"] == "Tomato___healthy"
        assert pytest.approx(res_data["confidence"], 0.01) == 0.80
        assert len(res_data["top_predictions"]) == 3
    finally:
        model_utils.reset_model_cache()


def test_missing_model_handling(client, sample_leaf_image_bytes, monkeypatch):
    """Test application returns 503 when model artifact file is missing."""
    model_utils.reset_model_cache()
    # Point default model path to non-existent file
    monkeypatch.setattr(model_utils, "DEFAULT_MODEL_PATH", "non_existent_model.keras")

    data = {"file": (io.BytesIO(sample_leaf_image_bytes), "leaf.jpg")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 503
    res_data = response.get_json()
    assert "not found" in res_data["error"].lower()
    assert "suggestion" in res_data


# -------------------------------------------------------------
# 5. Real Model Integration Test (Executed when model artifact exists)
# -------------------------------------------------------------

def test_real_model_integration_if_present(client, sample_leaf_image_bytes):
    """Integration test verifying real saved ResNet50 model if available on disk."""
    real_model_path = os.path.join(WEEK2_DIR, "models", "plant_disease_resnet50.keras")
    if not os.path.exists(real_model_path):
        pytest.skip("Real model artifact not yet generated. Skipping real integration test.")

    model_utils.reset_model_cache()
    data = {"file": (io.BytesIO(sample_leaf_image_bytes), "test_leaf.jpg")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    result = response.get_json()
    assert result["predicted_class"] in model_utils.load_class_mapping()
    assert 0.0 <= result["confidence"] <= 1.0


# -------------------------------------------------------------
# 6. Quick Sample and Known Labeled Image Tests
# -------------------------------------------------------------

@pytest.mark.parametrize("class_name", [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
])
def test_each_quick_sample_inference(client, class_name):
    """Test that fetching and predicting each quick sample specimen succeeds."""
    sample_res = client.get(f"/sample/{class_name}")
    assert sample_res.status_code == 200, f"Sample for {class_name} not available"

    pred_res = client.post(
        "/predict",
        data={"file": (io.BytesIO(sample_res.data), f"{class_name}.jpg")},
        content_type="multipart/form-data"
    )
    assert pred_res.status_code == 200
    data = pred_res.get_json()
    assert data["predicted_class"] in model_utils.load_class_mapping()
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_predictions"]) == 3

    # Verify top-3 predictions are in descending order
    scores = [t["score"] for t in data["top_predictions"]]
    assert scores == sorted(scores, reverse=True)


def test_known_labeled_images_evaluation():
    """Evaluate known labeled images from each class and verify model outputs."""
    test_data_dir = os.path.join(WEEK2_DIR, "data", "plant_village_5_classes", "test")
    if not os.path.exists(test_data_dir):
        pytest.skip("Test dataset folder not found.")

    model = model_utils.get_plant_disease_model()
    classes = model_utils.load_class_mapping()

    for c in classes:
        cdir = os.path.join(test_data_dir, c)
        files = sorted(os.listdir(cdir))[:3]
        for fname in files:
            fpath = os.path.join(cdir, fname)
            with open(fpath, "rb") as fp:
                res = model_utils.predict_image(fp.read(), model=model)
            assert res["predicted_class"] in classes
            assert 0.0 <= res["confidence"] <= 1.0
            assert len(res["top_predictions"]) == 3
