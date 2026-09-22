"""
ensemble.py — Weighted soft-voting ensemble for NeuroNova.

weighted_soft_vote() combines probability vectors from three CNN models
using the weights defined in ensemble_config.json and returns a structured
result dict ready to be serialised as JSON.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def ensure_probabilities(vec: np.ndarray) -> np.ndarray:
    """
    Guarantee the vector is a valid probability distribution.

    - If values sum close to 1.0 → assume they're already probabilities.
    - Otherwise → apply softmax to convert logits.
    - Either way, clip to [0, 1] and renormalise for safety.
    """
    vec = vec.astype(np.float64)

    total = float(vec.sum())
    if abs(total - 1.0) > 0.01:
        logger.debug("Output sum=%.4f — applying softmax to convert logits", total)
        vec = softmax(vec)
    else:
        # Already probabilities; normalise for floating-point precision
        vec = vec / total

    vec = np.clip(vec, 0.0, 1.0)
    return vec


def _model_summary(probs: np.ndarray, class_names: List[str]) -> dict:
    """Build per-model prediction summary from probability vector."""
    idx = int(np.argmax(probs))
    return {
        "prediction": class_names[idx],
        "confidence": float(probs[idx]),
        "confidence_percent": round(float(probs[idx]) * 100, 2),
        "probabilities": {
            name: float(probs[i]) for i, name in enumerate(class_names)
        },
    }


def weighted_soft_vote(
    mobilenet_probs: np.ndarray,
    resnet_probs: np.ndarray,
    vgg_probs: np.ndarray,
    weights: List[float],
    class_names: List[str],
    calibrate: bool = False,
    temperature: float = 0.45,
) -> dict:
    """
    Perform weighted soft-voting ensemble with optional entropy-adaptive calibration.

    Parameters
    ----------
    mobilenet_probs, resnet_probs, vgg_probs :
        Raw output vectors from each model (logits OR probabilities).
    weights : [w_mobilenet, w_resnet, w_vgg]
        Must sum to 1.0.
    class_names : ordered list of class labels.
    calibrate : bool
        If True, applies model entropy-weighted consensus, hierarchical pathology gating,
        and temperature scaling.
    temperature : float
        Softmax sharpening temperature (lower = more decisive separation).

    Returns
    -------
    dict with keys:
        prediction, probabilities, models, ensemble
    """
    n_classes = len(class_names)

    # Validate output lengths
    for name, vec in [
        ("MobileNetV2", mobilenet_probs),
        ("ResNet50", resnet_probs),
        ("VGG16", vgg_probs),
    ]:
        if len(vec) != n_classes:
            raise ValueError(
                f"{name} output length {len(vec)} != number of classes {n_classes}"
            )

    # Convert to clean probability distributions
    mob_p = ensure_probabilities(mobilenet_probs)
    res_p = ensure_probabilities(resnet_probs)
    vgg_p = ensure_probabilities(vgg_probs)

    w_mob, w_res, w_vgg = weights[0], weights[1], weights[2]

    if calibrate:
        # Entropy-adaptive dynamic weighting
        def _model_certainty(p: np.ndarray) -> float:
            p_act = p[p > 0]
            ent = -float(np.sum(p_act * np.log2(p_act)))
            max_ent = float(np.log2(n_classes))
            norm_ent = ent / max_ent if max_ent > 0 else 0.0
            return float((max(0.0, 1.0 - norm_ent)) ** 1.5)

        c_mob = _model_certainty(mob_p)
        c_res = _model_certainty(res_p)
        c_vgg = _model_certainty(vgg_p)

        # Modulate base weights with model-specific confidence
        w_mob_dyn = w_mob * (0.35 + 0.65 * c_mob)
        w_res_dyn = w_res * (0.35 + 0.65 * c_res)
        w_vgg_dyn = w_vgg * (0.35 + 0.65 * c_vgg)
        tot_w = w_mob_dyn + w_res_dyn + w_vgg_dyn
        w_mob, w_res, w_vgg = w_mob_dyn / tot_w, w_res_dyn / tot_w, w_vgg_dyn / tot_w

        consensus = w_mob * mob_p + w_res * res_p + w_vgg * vgg_p
        consensus = np.clip(consensus, 1e-7, 1.0)
        consensus /= consensus.sum()

        # Temperature scaling
        logits = np.log(consensus)
        ensemble_probs = np.exp((logits - np.max(logits)) / max(0.1, temperature))
        ensemble_probs /= ensemble_probs.sum()

        # HIERARCHICAL PATHOLOGY GATING:
        # Prevents the false-negative paradox where tumor probability is divided
        # between glioma and meningioma, allowing 'notumor' to win by mere plurality.
        notumor_idx = class_names.index("notumor") if "notumor" in class_names else -1
        if notumor_idx != -1:
            p_healthy = float(ensemble_probs[notumor_idx])
            p_tumor_total = 1.0 - p_healthy
            if p_tumor_total > p_healthy:
                # Pathology is Tumor: condition winning subtype strictly within tumor classes
                tumor_indices = [i for i in range(n_classes) if i != notumor_idx]
                tumor_subprobs = np.array([ensemble_probs[i] for i in tumor_indices])
                best_sub_idx = int(np.argmax(tumor_subprobs))
                predicted_idx = tumor_indices[best_sub_idx]
            else:
                predicted_idx = notumor_idx
        else:
            predicted_idx = int(np.argmax(ensemble_probs))
    else:
        # Standard weighted soft vote
        ensemble_probs = w_mob * mob_p + w_res * res_p + w_vgg * vgg_p
        ensemble_probs = ensemble_probs / ensemble_probs.sum()
        predicted_idx = int(np.argmax(ensemble_probs))

    predicted_class = class_names[predicted_idx]
    confidence = float(ensemble_probs[predicted_idx])

    logger.info(
        "Ensemble result: %s (%.2f%%)  [mob=%.3f res=%.3f vgg=%.3f]",
        predicted_class,
        confidence * 100,
        float(mob_p[predicted_idx]),
        float(res_p[predicted_idx]),
        float(vgg_p[predicted_idx]),
    )

    # High-value clinical and performance metrics
    probs_dict = {name: float(ensemble_probs[i]) for i, name in enumerate(class_names)}
    p_notumor = probs_dict.get("notumor", 0.0)
    p_tumor = max(0.0, min(1.0, 1.0 - p_notumor))

    # Normalized Shannon Entropy (0 = complete certainty, 1 = uniform confusion)
    active_probs = ensemble_probs[ensemble_probs > 0]
    shannon_entropy = -float(np.sum(active_probs * np.log2(active_probs)))
    max_entropy = float(np.log2(n_classes))
    norm_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
    certainty_score = round(max(0.0, min(100.0, (1.0 - norm_entropy) * 100.0)), 1)

    tumor_detected = predicted_class != "notumor"
    if not tumor_detected:
        risk_level = "Low Risk (Healthy / No Tumor)"
    elif confidence >= 0.85:
        risk_level = "High Probability Tumor"
    else:
        risk_level = "Moderate Probability Tumor"

    # Consensus metric
    model_preds = [
        class_names[int(np.argmax(mob_p))],
        class_names[int(np.argmax(res_p))],
        class_names[int(np.argmax(vgg_p))],
    ]
    agreement_count = sum(1 for p in model_preds if p == predicted_class)
    agreement_ratio = round((agreement_count / 3.0) * 100, 1)

    metrics = {
        "tumor_detected": tumor_detected,
        "tumor_probability_percent": round(p_tumor * 100, 2),
        "healthy_probability_percent": round(p_notumor * 100, 2),
        "certainty_index_percent": certainty_score,
        "model_agreement_percent": agreement_ratio,
        "risk_level": risk_level,
        "normalized_entropy": round(norm_entropy, 4),
    }

    return {
        "prediction": {
            "class": predicted_class,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
        },
        "probabilities": probs_dict,
        "metrics": metrics,
        "models": {
            "mobilenet": _model_summary(mob_p, class_names),
            "resnet50": _model_summary(res_p, class_names),
            "vgg16": _model_summary(vgg_p, class_names),
        },
        "ensemble": {
            "weights": {
                "mobilenet": w_mob,
                "resnet50": w_res,
                "vgg16": w_vgg,
            }
        },
    }
