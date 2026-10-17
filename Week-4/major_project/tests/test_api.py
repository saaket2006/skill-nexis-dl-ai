"""
Tests for FastAPI REST API endpoints using TestClient.
"""

import io
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Ensure backend imports work
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_test_image_bytes(format="PNG", size=(200, 200), color="white"):
    """Helper to generate in-memory dummy image."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert len(data["supported_classes"]) == 5
    assert "version" in data


def test_classes_endpoint(client):
    response = client.get("/classes")
    assert response.status_code == 200
    classes = response.json()
    assert len(classes) == 5
    assert set(classes) == {"form", "invoice", "letter", "receipt", "report"}


def test_predict_endpoint_valid_image(client):
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/predict",
        files={"file": ("test_doc.png", img_bytes, "image/png")},
        data={"top_k": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"] in {"form", "invoice", "letter", "receipt", "report"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_k"]) == 3
    assert data["inference_time_ms"] > 0
    assert data["filename"] == "test_doc.png"


def test_predict_endpoint_invalid_file_type(client):
    dummy_text = b"This is a text document, not an image."
    response = client.post(
        "/predict",
        files={"file": ("test.txt", dummy_text, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_predict_endpoint_corrupted_image(client):
    corrupted_bytes = b"GIF89a corrupted image stream\x00\x00\xff"
    response = client.post(
        "/predict",
        files={"file": ("bad_image.png", corrupted_bytes, "image/png")}
    )
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "cannot identify image file" in detail or "failed" in detail or "invalid" in detail


def test_predict_top_k_parameter(client):
    img_bytes = create_test_image_bytes("JPEG")
    response = client.post(
        "/predict",
        files={"file": ("doc.jpg", img_bytes, "image/jpeg")},
        data={"top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["top_k"]) == 5
    # Verify descending confidence order
    confs = [p["confidence"] for p in data["top_k"]]
    assert confs == sorted(confs, reverse=True)
