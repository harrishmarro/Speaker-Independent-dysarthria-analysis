import numpy as np
import pytest

from dysarthria.acoustic_cv import leave_one_speaker_out, predict_speakers


def test_speaker_cv_scores_each_training_speaker_once_without_self_training():
    values = {
        "C1": np.zeros(20), "C2": np.full(20, 0.1), "C3": np.full(20, 0.2),
        "D1": np.full(20, 4.0), "D2": np.full(20, 4.1), "D3": np.full(20, 4.2),
    }
    labels = {name: ("control" if name.startswith("C") else "dysarthria")
              for name in values}
    for classifier in ("centroid", "speaker_knn", "ridge"):
        settings = {"classifier": classifier, "spectral_weight": 1.0, "ridge_alpha": 1.0,
                    "neighbors": 1}
        rows = leave_one_speaker_out(values, labels, settings)
        assert {row["speaker_id"] for row in rows} == set(values)
        assert all(row["target"] == row["prediction"] for row in rows)


def test_target_speaker_cannot_be_in_training_set():
    features = {"C1": np.zeros(20), "D1": np.ones(20)}
    labels = {"C1": "control", "D1": "dysarthria"}
    with pytest.raises(ValueError, match="disjoint"):
        predict_speakers(features, labels, {"C1": features["C1"]},
                         {"classifier": "ridge", "ridge_alpha": 1.0,
                          "spectral_weight": 1.0})
