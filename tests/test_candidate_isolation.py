import csv
import sys

import pytest

import research
import train
from locked import candidate_io


def test_candidate_manifest_withholds_test_rows(tmp_path):
    prepared = tmp_path / "prepared.csv"
    with prepared.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audio_path", "speaker_id", "label", "split"))
        writer.writeheader()
        for speaker, split in (("A", "train"), ("B", "validation"), ("C", "test")):
            writer.writerow({"audio_path": f"{speaker}.wav", "speaker_id": speaker,
                             "label": "control", "split": split})
    destination = tmp_path / "candidate.csv"
    assert research.write_candidate_manifest(prepared, destination) == 1
    with destination.open(newline="") as handle:
        assert {row["split"] for row in csv.DictReader(handle)} == {"train", "validation"}


def test_train_refuses_manifest_with_test_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate_io, "ARTIFACTS", tmp_path)
    manifest = tmp_path / "prepared.csv"
    manifest.write_text("audio_path,speaker_id,label,split\nx.wav,T1,control,test\n")
    monkeypatch.setattr(sys, "argv", ["train.py", "--manifest", str(manifest),
                                      "--output", str(tmp_path / "p.jsonl")])
    with pytest.raises(ValueError, match="only train and validation"):
        train.main()
