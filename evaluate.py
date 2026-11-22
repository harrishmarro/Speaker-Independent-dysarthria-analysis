#!/usr/bin/env python3
"""Locked evaluation entry point for saved speaker-level predictions.

Ground truth and the expected speaker set come from the locked prepared manifest, never
from the candidate's prediction file. Any ``target`` field a candidate writes is ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dysarthria.metrics import binary_classification_metrics, roc_auc

SPLITS = {
    "train": {"train"},
    "validation": {"validation"},
    "development": {"train", "validation"},
    "test": {"test"},
}


def expected_labels(manifest: str | Path, split: str) -> dict[str, int]:
    """Return one locked binary label per speaker in the requested split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    labels: dict[str, int] = {}
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in SPLITS[split]:
                continue
            label = int(row["label"] == "dysarthria")
            if labels.setdefault(row["speaker_id"], label) != label:
                raise ValueError(f"conflicting labels for speaker {row['speaker_id']}")
    if not labels:
        raise ValueError(f"manifest has no speakers in split {split!r}")
    return labels


def evaluate(records: list[dict[str, object]], labels: dict[str, int]) -> dict[str, float | int]:
    if not records:
        raise ValueError("prediction file contains no records")
    speaker_ids = [str(record["speaker_id"]) for record in records]
    if len(speaker_ids) != len(set(speaker_ids)):
        raise ValueError("evaluation expects one aggregate prediction per speaker")
    missing = sorted(set(labels) - set(speaker_ids))
    unexpected = sorted(set(speaker_ids) - set(labels))
    if missing or unexpected:
        raise ValueError(
            f"prediction speakers do not match the locked split: "
            f"missing={missing} unexpected={unexpected}"
        )
    ordered = sorted(records, key=lambda record: str(record["speaker_id"]))
    targets = [labels[str(record["speaker_id"])] for record in ordered]
    metrics: dict[str, float | int] = dict(binary_classification_metrics(
        targets, [int(record["prediction"]) for record in ordered],  # type: ignore[call-overload]
    ))
    if all("probability" in record for record in ordered) and len(set(targets)) == 2:
        metrics["auc"] = roc_auc(
            targets, [float(record["probability"]) for record in ordered],  # type: ignore[arg-type]
        )
    metrics["speakers"] = len(ordered)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True, help="locked prepared manifest")
    parser.add_argument("--split", required=True, choices=sorted(SPLITS))
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    with Path(arguments.predictions).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    metrics = evaluate(records, expected_labels(arguments.manifest, arguments.split))
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
