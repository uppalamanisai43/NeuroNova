"""
conftest.py — shared pytest fixtures for NeuroNova backend tests.
"""

import io
import os
import sys
import json

import numpy as np
import pytest
from PIL import Image

# Put backend/ on path so test files can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_image_bytes():
    """Return valid JPEG bytes of a 224x224 white image."""
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_png_bytes():
    """Return valid PNG bytes of a 100x100 black image."""
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def class_names():
    return ["glioma", "meningioma", "notumor", "pituitary"]


@pytest.fixture
def weights():
    return [0.2, 0.6, 0.2]
