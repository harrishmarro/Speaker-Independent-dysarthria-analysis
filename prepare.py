#!/usr/bin/env python3
"""Locked preparation entry point: validate input and create speaker-wise splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from dysarthria.manifest import assign_speaker_splits, read_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/sample_manifest.csv")
    parser.add_argument("--output", default="artifacts/prepared_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    rows = read_manifest(arguments.manifest)
    prepared = assign_speaker_splits(rows, seed=arguments.seed)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(prepared, output)
    counts: dict[str, set[str]] = {}
    for row in prepared:
        counts.setdefault(row["split"], set()).add(row["speaker_id"])
    summary = ", ".join(
        f"{split}={len(speakers)} speakers" for split, speakers in sorted(counts.items())
    )
    print(f"prepared {len(prepared)} recordings: {summary}")


if __name__ == "__main__":
    main()
