import csv
import json
import sys

import numpy as np
import pytest
import soundfile as sf

import train
from dysarthria.mfcc_features import mfcc_summary
from locked import candidate_io


def test_mfcc_summary_is_finite_and_responds_to_frequency():
    time = np.arange(16000) / 16000
    low = mfcc_summary(np.sin(2 * np.pi * 200 * time), 16000)
    high = mfcc_summary(np.sin(2 * np.pi * 2000 * time), 16000)
    assert low.shape == (26,)
    assert np.isfinite(low).all()
    assert np.linalg.norm(low - high) > 1.0
    assert np.isfinite(mfcc_summary(np.zeros(16000), 16000)).all()


def test_mfcc_rejects_unexpected_sample_rate():
    with pytest.raises(ValueError, match="16 kHz"):
        mfcc_summary(np.zeros(16000), 8000)


def test_mfcc_candidate_preserves_spectral_baseline_cv(tmp_path, monkeypatch):
    root = tmp_path / "project"
    configs = root / "configs" / "research"
    configs.mkdir(parents=True)
    (configs / "active.json").write_text(json.dumps({
        "candidate": "mfcc_ridge_1", "feature_set": "mfcc", "classifier": "ridge",
        "ridge_alpha": 1.0, "spectral_weight": 1.0, "aggregation": "mean",
    }))
    (configs / "baseline.json").write_text(json.dumps({
        "candidate": "spectral_equal", "spectral_weight": 1.0, "aggregation": "mean",
    }))
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.setattr(candidate_io, "ARTIFACTS", artifact_dir)
    monkeypatch.setattr(train, "__file__", str(root / "train.py"))
    manifest = artifact_dir / "prepared.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audio_path", "speaker_id", "label", "split"))
        writer.writeheader()
        for index, (speaker, label, split) in enumerate((
            ("C1", "control", "train"), ("C2", "control", "train"),
            ("D1", "dysarthria", "train"), ("D2", "dysarthria", "train"),
            ("C3", "control", "validation"), ("D3", "dysarthria", "validation"),
        )):
            audio = tmp_path / f"{speaker}.wav"
            time = np.arange(3200) / 16000
            sf.write(audio, 0.2 * np.sin(2 * np.pi * (250 + index * 100) * time), 16000)
            writer.writerow({"audio_path": str(audio), "speaker_id": speaker,
                             "label": label, "split": split})
    output = artifact_dir / "predictions.jsonl"
    monkeypatch.setattr(sys, "argv", ["train.py", "--manifest", str(manifest),
                                      "--output", str(output)])
    train.main()
    assert len(output.read_text().splitlines()) == 2
    assert len((artifact_dir / "cv_predictions.jsonl").read_text().splitlines()) == 4
    assert len((artifact_dir / "baseline_cv_predictions.jsonl").read_text().splitlines()) == 4
