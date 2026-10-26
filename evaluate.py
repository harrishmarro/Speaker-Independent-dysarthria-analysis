#!/usr/bin/env python3
"""Locked evaluation entry point for saved validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dysarthria.metrics import binary_classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    with Path(arguments.predictions).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("prediction file contains no records")
    speaker_ids = [record["speaker_id"] for record in records]
    if len(speaker_ids) != len(set(speaker_ids)):
        raise ValueError("evaluation expects one aggregate prediction per speaker")
    metrics = binary_classification_metrics(
        [record["target"] for record in records],
        [record["prediction"] for record in records],
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
