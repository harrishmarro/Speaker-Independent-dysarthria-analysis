"""Probe score rankings against documented, speaker-level severity labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations, permutations
from math import factorial, isfinite, sqrt


def _pair_directions(values: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(
        int(first < second) - int(first > second) for first, second in combinations(values, 2)
    )


def kendall_tau_b(x: Iterable[float], y: Iterable[float]) -> float:
    """Compute Kendall's tau-b, with zero for an undefined coefficient."""
    x_values, y_values = tuple(x), tuple(y)
    if len(x_values) != len(y_values):
        raise ValueError("Kendall tau-b needs equal-length inputs")
    if not all(map(isfinite, x_values + y_values)):
        raise ValueError("Kendall tau-b needs finite values")

    x_pairs = _pair_directions(x_values)
    y_pairs = _pair_directions(y_values)
    x_untied = sum(map(abs, x_pairs))
    y_untied = sum(map(abs, y_pairs))
    denominator = sqrt(x_untied * y_untied)
    if denominator == 0:
        return 0.0
    numerator = sum(a * b for a, b in zip(x_pairs, y_pairs, strict=True))
    return numerator / denominator


def _exact_p(scores: tuple[float, ...], severity: tuple[int, ...]) -> float:
    # Orient non-tied score pairs from smaller to larger score. The tau-b
    # denominator is invariant to permutation, so compare integer numerators.
    pairs = [
        (left, right) if a < b else (right, left)
        for (left, a), (right, b) in combinations(enumerate(scores), 2)
        if a != b
    ]

    def concordance(labels: tuple[int, ...]) -> int:
        return sum(
            int(labels[left] < labels[right]) - int(labels[left] > labels[right])
            for left, right in pairs
        )

    observed = concordance(severity)
    extreme = sum(concordance(labels) >= observed for labels in permutations(severity))
    return extreme / factorial(len(severity))


def _ordered_scores(
    scores: dict[str, float], speakers: tuple[str, ...], name: str
) -> tuple[float, ...]:
    missing = sorted(set(speakers) - scores.keys())
    if missing:
        raise ValueError(f"{name} missing scores for labeled speakers: {', '.join(missing)}")
    return tuple(scores[speaker] for speaker in speakers)


def rank_probe(
    scores: dict[str, float],
    severity: Mapping[str, int | None],
    controls: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Test the positive association and check controls on the same labeled speakers.

    The caller supplies the dysarthric speakers eligible for the evaluation.
    Null severity labels are ignored. Every labeled speaker needs a candidate
    score and a score from each supplied control.
    """
    labeled = [(speaker, label) for speaker, label in sorted(severity.items()) if label is not None]
    n = len(labeled)
    if n < 5:
        raise ValueError("rank_probe requires at least 5 speakers with non-null severity")
    if n > 9:
        raise ValueError("exact permutation requires n <= 9; a Monte Carlo variant is out of scope")

    speakers, labels = zip(*labeled, strict=True)
    values = _ordered_scores(scores, speakers, "candidate")
    tau = kendall_tau_b(values, labels)
    control_taus = {
        name: abs(kendall_tau_b(_ordered_scores(control, speakers, f"control {name!r}"), labels))
        for name, control in controls.items()
    }
    loo_min = min(
        kendall_tau_b(values[:index] + values[index + 1 :], labels[:index] + labels[index + 1 :])
        for index in range(n)
    )
    p = _exact_p(values, labels)
    failures = []
    if tau <= 0:
        failures.append("tau must be positive")
    if p >= 0.05:
        failures.append("exact permutation p must be below 0.05")
    if control_taus and tau <= max(control_taus.values()):
        failures.append("tau must exceed the strongest control |tau|")
    if loo_min <= 0:
        failures.append("leave-one-speaker-out tau must be positive for every omission")

    return {
        "tau": tau,
        "p": p,
        "control_taus": control_taus,
        "loo_min": loo_min,
        "n": n,
        "verdict": "no severity claim" if failures else "pass",
        "reason": ", ".join(failures) if failures else "all severity criteria met",
    }
