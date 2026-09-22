"""
config.py — NeuroNova configuration loader.

Reads ensemble_config.json and class_names.json, validates them,
and exposes a typed Config object used throughout the application.
"""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Project root: one level above backend/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Config directory (config/ at project root)
CONFIG_DIR = os.environ.get(
    "NEURONOVA_CONFIG_DIR",
    os.path.join(_PROJECT_ROOT, "config"),
)

# Models directory — defaults to project root where the .keras files live
MODELS_DIR = os.environ.get(
    "NEURONOVA_MODELS_DIR",
    _PROJECT_ROOT,
)

# Uploads directory
UPLOADS_DIR = os.environ.get(
    "NEURONOVA_UPLOADS_DIR",
    os.path.join(_PROJECT_ROOT, "uploads"),
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    class_names: List[str]
    weights: List[float]
    image_size: List[int]          # [width, height]
    model_filenames: Dict[str, str]  # {"mobilenet": "mobilenet_finetuned.keras", ...}


# ---------------------------------------------------------------------------
# Loader & validator
# ---------------------------------------------------------------------------

def load_config() -> EnsembleConfig:
    """Load and validate ensemble configuration from JSON files."""
    ensemble_path = os.path.join(CONFIG_DIR, "ensemble_config.json")
    class_names_path = os.path.join(CONFIG_DIR, "class_names.json")

    if not os.path.isfile(ensemble_path):
        raise FileNotFoundError(
            f"ensemble_config.json not found at: {ensemble_path}"
        )
    if not os.path.isfile(class_names_path):
        raise FileNotFoundError(
            f"class_names.json not found at: {class_names_path}"
        )

    with open(ensemble_path, "r") as f:
        raw = json.load(f)

    class_names: List[str] = raw["class_names"]
    weights: List[float] = raw["weights"]
    image_size: List[int] = raw["image_size"]
    model_filenames: Dict[str, str] = raw.get("models", {
        "mobilenet": "mobilenet_finetuned.keras",
        "resnet50": "resnet50_finetuned.keras",
        "vgg16": "vgg16_finetuned.keras",
    })

    # Validate weight count matches model count
    model_count = len(model_filenames)
    if len(weights) != model_count:
        raise ValueError(
            f"ensemble_config.json has {len(weights)} weights but "
            f"{model_count} models. They must match."
        )

    # Validate weight sum ≈ 1.0
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Ensemble weights sum to {weight_sum:.6f}, expected 1.0."
        )

    # Validate image size
    if len(image_size) != 2 or any(s <= 0 for s in image_size):
        raise ValueError(f"Invalid image_size in ensemble_config.json: {image_size}")

    logger.info(
        "Config loaded — classes: %s | weights: %s | image_size: %s",
        class_names,
        weights,
        image_size,
    )

    return EnsembleConfig(
        class_names=class_names,
        weights=weights,
        image_size=image_size,
        model_filenames=model_filenames,
    )


_HF_CDN_BASE = "https://huggingface.co/spaces/mani4117/neuronova-backend/resolve/main"


def get_model_path(filename: str) -> str:
    """Return absolute path for a model filename inside MODELS_DIR, downloading if missing."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.isfile(path):
        url = f"{_HF_CDN_BASE}/{filename}"
        logger.info("Model %s not found at %s. Downloading from CDN: %s ...", filename, path, url)
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(path, "wb") as out_f:
                out_f.write(resp.read())
            logger.info("Downloaded %s successfully (%d bytes).", filename, os.path.getsize(path))
        except Exception as e:
            raise FileNotFoundError(
                f"Model file not found at {path} and failed to download from {url}: {e}"
            )
    return path


# Singleton config loaded once at import time
try:
    CONFIG: EnsembleConfig = load_config()
except Exception as _e:  # noqa: BLE001
    # Allow import to succeed; app.py will call load_config() and surface the error
    CONFIG = None  # type: ignore[assignment]
    logger.warning("Config not yet loaded: %s", _e)
