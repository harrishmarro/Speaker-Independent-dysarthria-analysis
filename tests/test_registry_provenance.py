import csv

import research


def test_registry_adds_provenance_columns_without_losing_old_rows(tmp_path, monkeypatch):
    registry = tmp_path / "registry.tsv"
    registry.write_text("experiment_id\tbalanced_accuracy\nold\t0.5\n", encoding="utf-8")
    monkeypatch.setattr(research, "REGISTRY", registry)

    research.append_registry(
        {"experiment_id": "new", "balanced_accuracy": "0.75", "source_commit": "abc123"}
    )

    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows == [
        {"experiment_id": "old", "balanced_accuracy": "0.5", "source_commit": ""},
        {"experiment_id": "new", "balanced_accuracy": "0.75", "source_commit": "abc123"},
    ]
