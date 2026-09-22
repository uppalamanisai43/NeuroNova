"""
test_ensemble.py — Unit tests for the weighted soft-voting ensemble.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ensemble import weighted_soft_vote, ensure_probabilities, softmax


# ---------------------------------------------------------------------------
# ensure_probabilities
# ---------------------------------------------------------------------------

def test_ensure_probabilities_already_normalized():
    vec = np.array([0.7, 0.1, 0.1, 0.1])
    out = ensure_probabilities(vec)
    assert abs(out.sum() - 1.0) < 1e-6
    assert out[0] == pytest.approx(0.7, abs=1e-5)


def test_ensure_probabilities_logits_converted():
    """Logits with sum >> 1 should be softmax-ed."""
    logits = np.array([3.0, 1.0, 0.5, 0.2])
    out = ensure_probabilities(logits)
    assert abs(out.sum() - 1.0) < 1e-6
    assert out[0] > out[1] > out[2] > out[3]


def test_ensure_probabilities_clipped():
    """Negative values should be clipped to 0."""
    vec = np.array([1.0, 0.0, 0.0, 0.0])
    out = ensure_probabilities(vec)
    assert all(v >= 0 for v in out)


# ---------------------------------------------------------------------------
# weighted_soft_vote
# ---------------------------------------------------------------------------

def test_weighted_soft_vote_correct_class(class_names, weights):
    """When all models agree on glioma the ensemble should return glioma."""
    mob = np.array([0.9, 0.05, 0.03, 0.02])
    res = np.array([0.88, 0.07, 0.03, 0.02])
    vgg = np.array([0.85, 0.08, 0.04, 0.03])

    result = weighted_soft_vote(mob, res, vgg, weights, class_names)

    assert result["prediction"]["class"] == "glioma"
    assert result["prediction"]["confidence"] > 0.85
    assert abs(result["prediction"]["confidence_percent"] - result["prediction"]["confidence"] * 100) < 0.01


def test_weighted_soft_vote_math_correct(class_names, weights):
    """Manually verify the weighted average."""
    mob = np.array([0.5, 0.2, 0.2, 0.1])
    res = np.array([0.3, 0.4, 0.2, 0.1])
    vgg = np.array([0.4, 0.3, 0.2, 0.1])

    # Expected: 0.2*mob + 0.6*res + 0.2*vgg
    expected_raw = 0.2 * mob + 0.6 * res + 0.2 * vgg
    expected = expected_raw / expected_raw.sum()

    result = weighted_soft_vote(mob, res, vgg, weights, class_names)

    for i, name in enumerate(class_names):
        assert result["probabilities"][name] == pytest.approx(expected[i], abs=1e-6)


def test_weighted_soft_vote_probabilities_sum_to_one(class_names, weights):
    mob = np.array([0.6, 0.2, 0.1, 0.1])
    res = np.array([0.5, 0.3, 0.1, 0.1])
    vgg = np.array([0.7, 0.1, 0.1, 0.1])

    result = weighted_soft_vote(mob, res, vgg, weights, class_names)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-6


def test_weighted_soft_vote_structure(class_names, weights):
    """Check that the response dict has all required keys."""
    mob = np.array([0.25, 0.25, 0.25, 0.25])
    res = np.array([0.25, 0.25, 0.25, 0.25])
    vgg = np.array([0.25, 0.25, 0.25, 0.25])

    result = weighted_soft_vote(mob, res, vgg, weights, class_names)

    assert "prediction" in result
    assert "probabilities" in result
    assert "models" in result
    assert "ensemble" in result

    pred = result["prediction"]
    assert "class" in pred
    assert "confidence" in pred
    assert "confidence_percent" in pred

    for model_key in ["mobilenet", "resnet50", "vgg16"]:
        assert model_key in result["models"]
        m = result["models"][model_key]
        assert "prediction" in m
        assert "confidence" in m
        assert "probabilities" in m

    ew = result["ensemble"]["weights"]
    assert ew["mobilenet"] == pytest.approx(0.2)
    assert ew["resnet50"] == pytest.approx(0.6)
    assert ew["vgg16"] == pytest.approx(0.2)


def test_weighted_soft_vote_wrong_output_length(class_names, weights):
    """Should raise ValueError when output length doesn't match class count."""
    mob = np.array([0.5, 0.3, 0.2])       # wrong: 3 instead of 4
    res = np.array([0.5, 0.2, 0.2, 0.1])
    vgg = np.array([0.5, 0.2, 0.2, 0.1])

    with pytest.raises(ValueError, match="MobileNetV2 output length"):
        weighted_soft_vote(mob, res, vgg, weights, class_names)


def test_weighted_soft_vote_logit_inputs(class_names, weights):
    """Logit inputs (not summing to 1) should still yield a valid result."""
    mob = np.array([5.0, 2.0, 1.0, 0.5])   # logits
    res = np.array([6.0, 1.5, 1.0, 0.5])
    vgg = np.array([4.5, 2.0, 1.5, 0.5])

    result = weighted_soft_vote(mob, res, vgg, weights, class_names)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-6
    assert result["prediction"]["class"] == "glioma"
