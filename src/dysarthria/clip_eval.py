"""Evaluate binary speaker classification from a fixed number of clips."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from statistics import mean

import numpy as np
from numpy.typing import NDArray


def _group_takes(
    records: list[dict[str, object]], labels: dict[str, int]
) -> dict[str, dict[object, list[float]]]:
    speakers: dict[str, dict[object, list[float]]] = {speaker: {} for speaker in labels}
    for record in records:
        speaker = record.get("speaker_id")
        if not isinstance(speaker, str) or speaker not in speakers:
            raise ValueError(f"unknown speaker_id: {speaker!r}")
        score = record.get("score")
        if not isinstance(score, Real) or not isfinite(score):
            raise ValueError(f"score for speaker {speaker!r} must be a finite number")
        take_id = record.get("take_id")
        # Each recording without provenance is treated as its own take.
        take_key = object() if take_id is None else take_id
        try:
            speakers[speaker].setdefault(take_key, []).append(float(score))
        except TypeError as error:
            raise ValueError("take_id must be hashable") from error
    return speakers


def _draw_scores(
    speakers: list[list[list[float]]], k: int, draws: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    scores = np.empty((draws, len(speakers)), dtype=np.float64)
    for draw in range(draws):
        for column, takes in enumerate(speakers):
            selected = rng.choice(len(takes), size=k, replace=False)
            scores[draw, column] = mean(
                takes[take][rng.integers(len(takes[take]))] for take in selected
            )
    return scores


def _speaker_bootstrap(
    positives: NDArray[np.float64],
    negatives: NDArray[np.float64],
    boots: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    # Averaging pairwise AUC credits and classification decisions over draws first
    # lets each bootstrap sample reuse the same speaker indices across all draws.
    auc_credit = (positives[:, :, None] > negatives[:, None, :]).mean(axis=0)
    auc_credit += 0.5 * (positives[:, :, None] == negatives[:, None, :]).mean(axis=0)
    sensitivity = (positives >= 0).mean(axis=0)
    specificity = (negatives < 0).mean(axis=0)
    positive_count, negative_count = auc_credit.shape
    aucs = np.empty(boots)
    balanced_accuracies = np.empty(boots)
    for index in range(boots):
        positive_ids = rng.integers(positive_count, size=(positive_count,))
        negative_ids = rng.integers(negative_count, size=(negative_count,))
        aucs[index] = auc_credit[np.ix_(positive_ids, negative_ids)].mean()
        balanced_accuracies[index] = (
            sensitivity[positive_ids].mean() + specificity[negative_ids].mean()
        ) / 2

    auc_lo, auc_hi = np.percentile(aucs, (2.5, 97.5))
    ba_lo, ba_hi = np.percentile(balanced_accuracies, (2.5, 97.5))
    return {
        "auc_mean": float(auc_credit.mean()),
        "auc_lo": float(auc_lo),
        "auc_hi": float(auc_hi),
        "ba_mean": float((sensitivity.mean() + specificity.mean()) / 2),
        "ba_lo": float(ba_lo),
        "ba_hi": float(ba_hi),
    }


def k_clip_curve(
    records: list[dict[str, object]],
    labels: dict[str, int],
    ks: tuple[int, ...] = (1, 3, 5),
    draws: int = 200,
    boots: int = 1000,
    seed: int = 0,
) -> list[dict[str, float | int]]:
    """Average speaker metrics over clip draws and bootstrap speakers within class.

    Draw takes uniformly without replacement, then draw one recording uniformly
    from each selected take. A missing or null take ID gives that recording its own
    take. A score of zero is classified as positive.

    Each bootstrap replicate resamples speaker indices once for all draws. Speakers
    listed in labels with fewer than k takes are counted as excluded, and both
    classes must remain represented for each requested k.
    """
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    if any(not isinstance(count, int) or count < 1 for count in (draws, boots)):
        raise ValueError("draws and boots must be positive integers")
    if set(labels.values()) != {0, 1}:
        raise ValueError("labels must contain both classes, encoded as 0 and 1")

    grouped = _group_takes(records, labels)
    draw_rng, bootstrap_rng = np.random.default_rng(seed).spawn(2)
    rows: list[dict[str, float | int]] = []
    for k in ks:
        by_class: dict[int, list[list[list[float]]]] = {0: [], 1: []}
        for speaker, takes in sorted(grouped.items()):
            if len(takes) >= k:
                by_class[labels[speaker]].append(list(takes.values()))
        if not by_class[0] or not by_class[1]:
            raise ValueError(f"k={k} needs eligible speakers from both classes")
        speakers = len(by_class[0]) + len(by_class[1])
        positives = _draw_scores(by_class[1], k, draws, draw_rng)
        negatives = _draw_scores(by_class[0], k, draws, draw_rng)
        rows.append(
            {
                "k": k,
                "speakers": speakers,
                "excluded": len(labels) - speakers,
                **_speaker_bootstrap(positives, negatives, boots, bootstrap_rng),
            }
        )
    return rows
