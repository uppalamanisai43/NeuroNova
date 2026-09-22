"""
app.py — NeuroNova Flask application.

Endpoints:
  GET  /            → backend info
  GET  /health      → health check
  GET  /model-info  → model metadata
  POST /predict     → MRI image classification

Models are loaded ONCE at startup via _load_services().
During tests, set NEURONOVA_TESTING=1 to skip auto-loading.
"""

import io
import logging
import os
import sys
import time

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")
CORS(app, origins=_CORS_ORIGINS, supports_credentials=False)

# ---------------------------------------------------------------------------
# Module-level service references (populated by _load_services())
# ---------------------------------------------------------------------------
cfg = None           # EnsembleConfig
model_service = None  # ModelService


def _load_services():
    """Load config and models. Called once at startup."""
    global cfg, model_service

    from config import load_config
    from model_service import ModelService

    logger.info("NeuroNova backend starting …")

    try:
        cfg = load_config()
    except Exception as e:
        logger.critical("FATAL — Failed to load configuration: %s", e)
        sys.exit(1)

    try:
        model_service = ModelService(cfg)
    except Exception as e:
        logger.critical("FATAL — Failed to load models: %s", e)
        sys.exit(1)

    logger.info("Application started. Models loaded successfully.")


# Only auto-load when NOT in test mode
if os.environ.get("NEURONOVA_TESTING") != "1":
    _load_services()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_extension(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS
    )


def _error(message: str, status: int = 400) -> tuple:
    return jsonify({"success": False, "error": message}), status


def _open_image_safely(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc
    img = Image.open(io.BytesIO(data))
    return img


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "NeuroNova Brain MRI Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health":     "GET /health",
            "model_info": "GET /model-info",
            "predict":    "POST /predict",
        },
        "disclaimer": (
            "For educational and research use only. "
            "Not a medical diagnostic device."
        ),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": model_service.is_loaded if model_service else False,
        "num_classes": len(cfg.class_names) if cfg else 0,
        "class_names": cfg.class_names if cfg else [],
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    try:
        info = model_service.model_info()
        return jsonify({"success": True, **info})
    except Exception as e:
        logger.exception("Error fetching model info")
        return _error("Unable to retrieve model information.", 500)


@app.route("/predict", methods=["POST"])
def predict():
    from ensemble import weighted_soft_vote

    logger.info("Prediction request received")

    if "image" not in request.files:
        return _error("No image file provided. Send a multipart/form-data request with key 'image'.")

    file = request.files["image"]

    if not file.filename:
        return _error("No filename in uploaded file.")

    if not _allowed_extension(file.filename):
        return _error(
            "Unsupported image format. Please upload a JPG, JPEG, or PNG file."
        )

    try:
        image_bytes = file.read()
    except Exception:
        logger.exception("Failed to read uploaded file bytes")
        return _error("Failed to read uploaded file.")

    if len(image_bytes) == 0:
        return _error("Uploaded file is empty.")

    try:
        pil_image = _open_image_safely(image_bytes)
    except ValueError as exc:
        logger.warning("Image validation failed: %s", exc)
        return _error(
            "The uploaded image could not be read. Please upload a valid image file."
        )

    try:
        uploads_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        pil_image.save(os.path.join(uploads_dir, "latest_scan.png"))
    except Exception:
        pass

    logger.info(
        "Image preprocessing completed — original size: %s, mode: %s",
        pil_image.size,
        pil_image.mode,
    )

    t_start = time.perf_counter()
    try:
        mob_probs, res_probs, vgg_probs = model_service.predict(pil_image)
    except Exception:
        logger.exception("Model inference failed")
        return _error("An error occurred during model inference. Please try again.", 500)

    logger.info("Computing weighted soft-vote ensemble …")
    try:
        result = weighted_soft_vote(
            mobilenet_probs=mob_probs,
            resnet_probs=res_probs,
            vgg_probs=vgg_probs,
            weights=cfg.weights,
            class_names=cfg.class_names,
            calibrate=True,
        )
    except Exception:
        logger.exception("Ensemble voting failed")
        return _error("Ensemble computation failed.", 500)

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 1)

    scan_meta = getattr(model_service, "last_scan_meta", None)
    if not isinstance(scan_meta, dict):
        scan_meta = {}

    if "metrics" in result:
        result["metrics"]["latency_ms"] = latency_ms
        result["metrics"]["scan_metadata"] = scan_meta

    logger.info(
        "Prediction returned: %s (%.2f%%) in %.1f ms",
        result["prediction"]["class"],
        result["prediction"]["confidence_percent"],
        latency_ms,
    )

    return jsonify({"success": True, **result})


# ---------------------------------------------------------------------------
# Global error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(e):
    return _error("Image exceeds the maximum allowed size of 10 MB.", 413)


@app.errorhandler(404)
def not_found(e):
    return _error("Endpoint not found.", 404)


@app.errorhandler(405)
def method_not_allowed(e):
    return _error("Method not allowed.", 405)


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled internal server error")
    return _error("Internal server error.", 500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting Flask server on port %d (debug=%s) …", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
