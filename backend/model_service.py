"""
model_service.py — NeuroNova model loading and inference service.

ModelService loads all three .keras models ONCE at startup.
It inspects each model for embedded preprocessing layers so that
preprocessing.py can apply the correct external preprocessing (or skip it).

Usage
-----
    service = ModelService(config)
    result = service.predict(pil_image)
"""

import logging
import os
import time
from typing import Tuple

import numpy as np
from PIL import Image

from config import EnsembleConfig, get_model_path
from preprocessing import (
    clean_mri_image,
    get_tta_variants,
    preprocess_mobilenet_batch,
    preprocess_resnet_batch,
    preprocess_vgg_batch,
    model_has_preprocessing_layer,
)

logger = logging.getLogger(__name__)


class ModelService:
    """
    Loads and holds references to all three trained Keras models.
    Exposes predict() which runs all three models and returns raw probability vectors.
    """

    def __init__(self, cfg: EnsembleConfig) -> None:
        self.cfg = cfg
        self._loaded = False

        # Will be set during load()
        self.mobilenet = None
        self.resnet50 = None
        self.vgg16 = None

        # Preprocessing flags
        self._mob_has_preproc = False
        self._res_has_preproc = False
        self._vgg_has_preproc = False

        self._load_all()

    def _load_all(self) -> None:
        """Load all three models from disk. Called once at startup."""
        import tensorflow as tf

        logger.info("=" * 60)
        logger.info("NeuroNova — Loading models …")

        target_size = tuple(self.cfg.image_size)  # e.g. (224, 224)

        # ---------- MobileNetV2 ----------
        mob_filename = self.cfg.model_filenames["mobilenet"]
        mob_path = get_model_path(mob_filename)
        logger.info("Loading MobileNetV2 from: %s", mob_path)
        t0 = time.perf_counter()
        self.mobilenet = tf.keras.models.load_model(mob_path, compile=False)
        t1 = time.perf_counter()
        self._mob_has_preproc = model_has_preprocessing_layer(self.mobilenet)
        logger.info(
            "MobileNetV2 loaded in %.2fs | embedded_preprocessing=%s | "
            "input_shape=%s | output_shape=%s",
            t1 - t0,
            self._mob_has_preproc,
            self.mobilenet.input_shape,
            self.mobilenet.output_shape,
        )

        # ---------- ResNet50 ----------
        res_filename = self.cfg.model_filenames["resnet50"]
        res_path = get_model_path(res_filename)
        logger.info("Loading ResNet50 from: %s", res_path)
        t0 = time.perf_counter()
        self.resnet50 = tf.keras.models.load_model(res_path, compile=False)
        t1 = time.perf_counter()
        self._res_has_preproc = model_has_preprocessing_layer(self.resnet50)
        logger.info(
            "ResNet50 loaded in %.2fs | embedded_preprocessing=%s | "
            "input_shape=%s | output_shape=%s",
            t1 - t0,
            self._res_has_preproc,
            self.resnet50.input_shape,
            self.resnet50.output_shape,
        )

        # ---------- VGG16 ----------
        vgg_filename = self.cfg.model_filenames["vgg16"]
        vgg_path = get_model_path(vgg_filename)
        logger.info("Loading VGG16 from: %s", vgg_path)
        t0 = time.perf_counter()
        self.vgg16 = tf.keras.models.load_model(vgg_path, compile=False)
        t1 = time.perf_counter()
        self._vgg_has_preproc = model_has_preprocessing_layer(self.vgg16)
        logger.info(
            "VGG16 loaded in %.2fs | embedded_preprocessing=%s | "
            "input_shape=%s | output_shape=%s",
            t1 - t0,
            self._vgg_has_preproc,
            self.vgg16.input_shape,
            self.vgg16.output_shape,
        )

        self._loaded = True
        logger.info("All three models loaded successfully.")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    def predict(self, pil_image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run all three models on a PIL image with Test-Time Augmentation (TTA).

        Returns
        -------
        (mobilenet_probs, resnet50_probs, vgg16_probs)
            Each is a 1-D numpy array of length = number of classes.
        """
        if not self._loaded:
            raise RuntimeError("Models are not loaded.")

        target_size = tuple(self.cfg.image_size)

        # 1. Clean MRI image (alpha channel, background color, ROI isolation, contrast)
        clean_img, meta = clean_mri_image(pil_image)
        self.last_scan_meta = meta

        # 2. Generate TTA multi-view variants for accuracy enhancement
        tta_variants = get_tta_variants(clean_img, target_size)
        logger.info(
            "Generated %d TTA views (ROI coverage: %.1f%%, cropped: %s, contrast: %s)",
            len(tta_variants),
            meta.get("roi_coverage_percent", 100.0),
            meta.get("cropped_roi", False),
            meta.get("contrast_enhanced", False),
        )

        # 3. Model-specific batch preprocessing
        mob_batch = preprocess_mobilenet_batch(tta_variants, self._mob_has_preproc)
        res_batch = preprocess_resnet_batch(tta_variants, self._res_has_preproc)
        vgg_batch = preprocess_vgg_batch(tta_variants, self._vgg_has_preproc)

        # 4. Inference & TTA averaging
        logger.info("Running MobileNetV2 inference …")
        mob_out = self.mobilenet.predict(mob_batch, verbose=0)
        mob_probs = np.mean(mob_out, axis=0).astype(np.float64)
        logger.info("MobileNetV2 prediction completed — top class idx: %d", int(np.argmax(mob_probs)))

        logger.info("Running ResNet50 inference …")
        res_out = self.resnet50.predict(res_batch, verbose=0)
        res_probs = np.mean(res_out, axis=0).astype(np.float64)
        logger.info("ResNet50 prediction completed — top class idx: %d", int(np.argmax(res_probs)))

        logger.info("Running VGG16 inference …")
        vgg_out = self.vgg16.predict(vgg_batch, verbose=0)
        vgg_probs = np.mean(vgg_out, axis=0).astype(np.float64)
        logger.info("VGG16 prediction completed — top class idx: %d", int(np.argmax(vgg_probs)))

        return mob_probs, res_probs, vgg_probs

    # ------------------------------------------------------------------
    def model_info(self) -> dict:
        """Return safe metadata about the loaded models (no filesystem paths)."""
        return {
            "models": {
                "mobilenet": {
                    "name": "MobileNetV2",
                    "filename": self.cfg.model_filenames["mobilenet"],
                    "input_shape": list(self.mobilenet.input_shape) if self.mobilenet else None,
                    "output_shape": list(self.mobilenet.output_shape) if self.mobilenet else None,
                    "embedded_preprocessing": self._mob_has_preproc,
                },
                "resnet50": {
                    "name": "ResNet50",
                    "filename": self.cfg.model_filenames["resnet50"],
                    "input_shape": list(self.resnet50.input_shape) if self.resnet50 else None,
                    "output_shape": list(self.resnet50.output_shape) if self.resnet50 else None,
                    "embedded_preprocessing": self._res_has_preproc,
                },
                "vgg16": {
                    "name": "VGG16",
                    "filename": self.cfg.model_filenames["vgg16"],
                    "input_shape": list(self.vgg16.input_shape) if self.vgg16 else None,
                    "output_shape": list(self.vgg16.output_shape) if self.vgg16 else None,
                    "embedded_preprocessing": self._vgg_has_preproc,
                },
            },
            "class_names": self.cfg.class_names,
            "num_classes": len(self.cfg.class_names),
            "image_size": self.cfg.image_size,
            "ensemble_weights": {
                "mobilenet": self.cfg.weights[0],
                "resnet50": self.cfg.weights[1],
                "vgg16": self.cfg.weights[2],
            },
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded
