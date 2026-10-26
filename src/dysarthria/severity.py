"""Experimental fusion helpers; not a clinically validated severity scale."""

from __future__ import annotations


def fused_score(dysarthria_probability: float, word_error_rate: float, alpha: float = 0.7) -> float:
    for name, value in {
        "dysarthria_probability": dysarthria_probability,
        "word_error_rate": word_error_rate,
        "alpha": alpha,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    return alpha * dysarthria_probability + (1 - alpha) * word_error_rate


def experimental_band(score: float) -> str:
    """Map a fused score to exploratory bands that require clinical calibration."""
    if not 0 <= score <= 1:
        raise ValueError("score must be between 0 and 1")
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "moderate"
    if score < 0.75:
        return "high"
    return "very_high"
