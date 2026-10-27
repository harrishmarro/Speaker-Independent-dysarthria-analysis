"""Advance a bounded, reviewable sequence of acoustic candidates.

This controller reads aggregate validation metrics only. It cannot access audio,
predictions, test labels, secrets, or model code.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "configs/research/active.json"
REGISTRY = ROOT / "experiments/registry.tsv"
CANDIDATES = (
    {"candidate": "spectral_equal", "spectral_weight": 1.0, "aggregation": "mean"},
    {"candidate": "spectral_stronger", "spectral_weight": 2.0, "aggregation": "mean"},
    {"candidate": "spectral_weaker", "spectral_weight": 0.5, "aggregation": "mean"},
    {"candidate": "spectral_robust", "spectral_weight": 1.0, "aggregation": "median"},
)


def next_candidate(
    active: dict[str, object], records: list[dict[str, str]]
) -> dict[str, object] | None:
    """Advance only after a real result; stop on a promising result or exhausted plan."""
    if active not in CANDIDATES:
        raise ValueError("active candidate is outside the reviewed plan")
    completed = [
        row for row in records
        if row["mode"] == "experiment"
        and row["status"] == "candidate_result"
        and row["description"] == active["candidate"]
    ]
    if not completed:
        return None
    if float(completed[-1]["balanced_accuracy"]) >= 0.75:
        return None  # Requires human review and repeated-speaker confirmation.
    index = CANDIDATES.index(active)
    return dict(CANDIDATES[index + 1]) if index + 1 < len(CANDIDATES) else None


def main() -> None:
    active = json.loads(ACTIVE.read_text(encoding="utf-8"))
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    proposed = next_candidate(active, records)
    if proposed is None:
        print("no eligible next candidate; retain current configuration for review")
        return
    ACTIVE.write_text(json.dumps(proposed, indent=2) + "\n", encoding="utf-8")
    print(f"proposed next candidate: {proposed['candidate']}")


if __name__ == "__main__":
    main()
