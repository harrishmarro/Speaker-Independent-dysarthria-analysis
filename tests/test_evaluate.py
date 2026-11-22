import csv

import pytest

import evaluate
from dysarthria.metrics import roc_auc


def write_manifest(path):
    rows = [("C1", "control", "train"), ("D1", "dysarthria", "train"),
            ("C2", "control", "validation"), ("D2", "dysarthria", "validation"),
            ("C3", "control", "test"), ("D3", "dysarthria", "test")]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audio_path", "speaker_id", "label", "split"))
        writer.writeheader()
        for speaker, label, split in rows:
            for take in range(2):
                writer.writerow({"audio_path": f"{speaker}_{take}.wav", "speaker_id": speaker,
                                 "label": label, "split": split})
    return path


def test_labels_come_from_locked_manifest(tmp_path):
    labels = evaluate.expected_labels(write_manifest(tmp_path / "m.csv"), "validation")
    assert labels == {"C2": 0, "D2": 1}
    development = evaluate.expected_labels(tmp_path / "m.csv", "development")
    assert set(development) == {"C1", "D1", "C2", "D2"}


def test_candidate_target_field_is_ignored():
    # A candidate claiming its wrong answers are correct must still score zero.
    records = [{"speaker_id": "C2", "prediction": 1, "target": 1},
               {"speaker_id": "D2", "prediction": 0, "target": 0}]
    metrics = evaluate.evaluate(records, {"C2": 0, "D2": 1})
    assert metrics["balanced_accuracy"] == 0.0


@pytest.mark.parametrize("speakers", [["C2"], ["C2", "D2", "C3"], ["C2", "X9"]])
def test_speaker_set_must_match_split_exactly(speakers):
    records = [{"speaker_id": s, "prediction": 0} for s in speakers]
    with pytest.raises(ValueError, match="do not match"):
        evaluate.evaluate(records, {"C2": 0, "D2": 1})


def test_auc_reported_from_probabilities():
    records = [{"speaker_id": "C2", "prediction": 0, "probability": 0.2},
               {"speaker_id": "D2", "prediction": 0, "probability": 0.4}]
    metrics = evaluate.evaluate(records, {"C2": 0, "D2": 1})
    assert metrics["auc"] == 1.0 and metrics["balanced_accuracy"] == 0.5


def test_roc_auc_handles_ties():
    assert roc_auc([0, 1, 0, 1], [0.1, 0.9, 0.5, 0.5]) == 0.875
