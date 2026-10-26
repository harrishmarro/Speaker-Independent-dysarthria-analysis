#!/usr/bin/env python3
"""Candidate experiment surface.

The initial implementation is a data-free smoke baseline. It predicts the majority training
label and proves that the locked research harness works. It is not a scientific model.
"""

from __future__ import annotations

import argparse
from collections import Counter

from locked.candidate_io import read_prepared_manifest, write_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    rows = read_prepared_manifest(arguments.manifest)
    training_labels = [row["label"] for row in rows if row["split"] == "train"]
    majority = Counter(training_labels).most_common(1)[0][0]
    probability = 1.0 if majority == "dysarthria" else 0.0

    predictions = []
    for row in rows:
        if row["split"] == "validation":
            predictions.append(
                {
                "speaker_id": row["speaker_id"],
                "target": 1 if row["label"] == "dysarthria" else 0,
                "prediction": 1 if majority == "dysarthria" else 0,
                "probability": probability,
                }
            )
    write_predictions(predictions, arguments.output)
    print(f"smoke baseline wrote validation predictions to {arguments.output}")


if __name__ == "__main__":
    main()
