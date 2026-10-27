"""Speaker-level acoustic predictions with fold-local preprocessing."""

from __future__ import annotations

import numpy as np


def predict_speakers(
    training: dict[str, np.ndarray],
    labels: dict[str, str],
    targets: dict[str, np.ndarray],
    settings: dict[str, object],
) -> list[dict[str, object]]:
    """Fit on training speakers only and predict one result per target speaker."""
    if not training or set(training) & set(targets):
        raise ValueError("training and target speakers must be nonempty and disjoint")
    if not set(training | targets) <= set(labels) or {labels[s] for s in training} != {
        "control", "dysarthria"
    }:
        raise ValueError("training speakers require known labels from both classes")
    weight = settings.get("spectral_weight")
    if weight not in (0.5, 1.0, 2.0):
        raise ValueError("unsupported spectral weight")
    classifier = settings.get("classifier", "centroid")
    if classifier not in ("centroid", "speaker_knn", "ridge"):
        raise ValueError("unsupported classifier")
    neighbors = settings.get("neighbors", 1)
    if neighbors not in (1, 3, 5):
        raise ValueError("unsupported neighbor count")
    ridge_alpha = settings.get("ridge_alpha", 1.0)
    if classifier == "ridge" and ridge_alpha not in (0.1, 1.0, 10.0):
        raise ValueError("unsupported ridge penalty")
    if not isinstance(ridge_alpha, (int, float)):
        raise ValueError("ridge penalty must be numeric")

    matrix = np.stack([training[speaker] for speaker in sorted(training)])
    mean = matrix.mean(axis=0)
    scale = np.where(matrix.std(axis=0) > 1e-6, matrix.std(axis=0), 1.0)
    weights = np.ones(matrix.shape[1], dtype=np.float32)
    weights[8:] = float(weight)

    def standardize(vector: np.ndarray) -> np.ndarray:
        return ((vector - mean) / scale) * weights

    fitted = [(speaker, labels[speaker], standardize(training[speaker]))
              for speaker in sorted(training)]
    centers = {
        label: np.mean([vector for _, item_label, vector in fitted if item_label == label], axis=0)
        for label in ("control", "dysarthria")
    }
    if classifier == "ridge":
        design = np.stack([np.concatenate((np.ones(1), vector)) for _, _, vector in fitted])
        response = np.array([1.0 if label == "dysarthria" else -1.0
                             for _, label, _ in fitted])
        counts = {label: sum(item_label == label for _, item_label, _ in fitted)
                  for label in ("control", "dysarthria")}
        sample_weights = np.array([len(fitted) / (2 * counts[label])
                                   for _, label, _ in fitted])
        penalty = np.eye(design.shape[1]) * float(ridge_alpha)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ (design * sample_weights[:, None]) + penalty,
            design.T @ (response * sample_weights),
        )
    predictions = []
    for speaker, raw_vector in sorted(targets.items()):
        vector = standardize(raw_vector)
        if classifier == "speaker_knn":
            closest = sorted(
                (float(np.linalg.norm(vector - other)), other_speaker, label)
                for other_speaker, label, other in fitted
            )[:int(neighbors)]
            probability = sum(label == "dysarthria" for _, _, label in closest) / len(closest)
        elif classifier == "ridge":
            margin = float(np.concatenate((np.ones(1), vector)) @ coefficients)
            probability = float(1.0 / (1.0 + np.exp(-margin)))
        else:
            margin = (float(np.linalg.norm(vector - centers["control"]))
                      - float(np.linalg.norm(vector - centers["dysarthria"])))
            probability = float(1.0 / (1.0 + np.exp(-margin)))
        predictions.append({
            "speaker_id": speaker,
            "target": int(labels[speaker] == "dysarthria"),
            "prediction": int(probability > 0.5),
            "probability": probability,
        })
    return predictions


def leave_one_speaker_out(
    features: dict[str, np.ndarray], labels: dict[str, str], settings: dict[str, object]
) -> list[dict[str, object]]:
    """Refit every fold, excluding the target speaker from scaling and model fitting."""
    results = []
    for speaker in sorted(features):
        training = {name: vector for name, vector in features.items() if name != speaker}
        results.extend(predict_speakers(
            training, labels, {speaker: features[speaker]}, settings,
        ))
    return results
