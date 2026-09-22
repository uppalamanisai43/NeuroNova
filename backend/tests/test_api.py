"""
test_api.py — API integration tests for NeuroNova Flask backend.

Sets NEURONOVA_TESTING=1 before importing app so that _load_services()
is skipped, then directly injects a mock model_service.
"""

import importlib
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock

# Ensure backend/ is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set testing flag BEFORE importing app (prevents model loading)
os.environ["NEURONOVA_TESTING"] = "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_image_bytes(fmt="JPEG", size=(224, 224), color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def _make_mock_service():
    svc = MagicMock()
    svc.is_loaded = True
    svc.predict.return_value = (
        np.array([0.90, 0.05, 0.03, 0.02]),   # mobilenet
        np.array([0.92, 0.04, 0.02, 0.02]),   # resnet50
        np.array([0.88, 0.06, 0.04, 0.02]),   # vgg16
    )
    svc.model_info.return_value = {
        "models": {},
        "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
        "num_classes": 4,
        "image_size": [224, 224],
        "ensemble_weights": {"mobilenet": 0.2, "resnet50": 0.6, "vgg16": 0.2},
    }
    return svc


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    import app as flask_app
    from config import EnsembleConfig

    # Inject mock service and config
    flask_app.model_service = _make_mock_service()
    flask_app.cfg = EnsembleConfig(
        class_names=["glioma", "meningioma", "notumor", "pituitary"],
        weights=[0.2, 0.6, 0.2],
        image_size=[224, 224],
        model_filenames={
            "mobilenet": "mobilenet_finetuned.keras",
            "resnet50": "resnet50_finetuned.keras",
            "vgg16": "vgg16_finetuned.keras",
        },
    )

    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c, flask_app.model_service


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

def test_index_returns_200(client):
    c, _ = client
    rv = c.get("/")
    assert rv.status_code == 200
    data = rv.get_json()
    assert "service" in data
    assert "NeuroNova" in data["service"]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    c, _ = client
    rv = c.get("/health")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------

def test_model_info_ok(client):
    c, _ = client
    rv = c.get("/model-info")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert "class_names" in data


# ---------------------------------------------------------------------------
# POST /predict — success
# ---------------------------------------------------------------------------

def test_predict_jpeg_success(client):
    c, _ = client
    img_bytes = make_image_bytes("JPEG")
    rv = c.post(
        "/predict",
        data={"image": (io.BytesIO(img_bytes), "mri.jpg")},
        content_type="multipart/form-data",
    )
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert "prediction" in data
    assert "probabilities" in data
    assert "models" in data
    assert "ensemble" in data
    assert data["prediction"]["class"] in ["glioma", "meningioma", "notumor", "pituitary"]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0


def test_predict_png_success(client):
    c, _ = client
    img_bytes = make_image_bytes("PNG")
    rv = c.post(
        "/predict",
        data={"image": (io.BytesIO(img_bytes), "mri.png")},
        content_type="multipart/form-data",
    )
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True


def test_predict_probabilities_sum_to_one(client):
    c, _ = client
    img_bytes = make_image_bytes("JPEG")
    rv = c.post(
        "/predict",
        data={"image": (io.BytesIO(img_bytes), "test.jpg")},
        content_type="multipart/form-data",
    )
    data = rv.get_json()
    total = sum(data["probabilities"].values())
    assert abs(total - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# POST /predict — error cases
# ---------------------------------------------------------------------------

def test_predict_no_file(client):
    c, _ = client
    rv = c.post("/predict", data={}, content_type="multipart/form-data")
    assert rv.status_code == 400
    data = rv.get_json()
    assert data["success"] is False


def test_predict_invalid_extension(client):
    c, _ = client
    rv = c.post(
        "/predict",
        data={"image": (io.BytesIO(b"fake data"), "mri.bmp")},
        content_type="multipart/form-data",
    )
    assert rv.status_code == 400
    data = rv.get_json()
    assert data["success"] is False
    assert "nsupp" in data["error"].lower() or "format" in data["error"].lower()


def test_predict_corrupted_image(client):
    c, _ = client
    rv = c.post(
        "/predict",
        data={"image": (io.BytesIO(b"this is not an image"), "mri.jpg")},
        content_type="multipart/form-data",
    )
    assert rv.status_code == 400
    data = rv.get_json()
    assert data["success"] is False


def test_predict_missing_filename(client):
    c, _ = client
    rv = c.post(
        "/predict",
        data={"image": (io.BytesIO(b"x"), "")},
        content_type="multipart/form-data",
    )
    assert rv.status_code == 400


def test_predict_empty_file(client):
    c, _ = client
    rv = c.post(
        "/predict",
        data={"image": (io.BytesIO(b""), "mri.jpg")},
        content_type="multipart/form-data",
    )
    assert rv.status_code == 400
    data = rv.get_json()
    assert data["success"] is False


# ---------------------------------------------------------------------------
# 404 / 405
# ---------------------------------------------------------------------------

def test_404(client):
    c, _ = client
    rv = c.get("/nonexistent")
    assert rv.status_code == 404


def test_method_not_allowed(client):
    c, _ = client
    rv = c.get("/predict")
    assert rv.status_code == 405
