"""Constrained file access exposed to autonomous candidate code."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (ROOT / "artifacts").resolve()


def _artifact_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ARTIFACTS):
        raise ValueError(f"candidate path must remain within {ARTIFACTS}")
    return resolved


def read_prepared_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest = _artifact_path(path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_predictions(records: list[dict[str, object]], path: str | Path) -> None:
    output = _artifact_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
