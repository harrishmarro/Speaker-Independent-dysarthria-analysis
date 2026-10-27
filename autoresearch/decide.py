"""Select and assess bounded research variants without reading private data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "configs/research/active.json"
ACCEPTED = ROOT / "configs/research/baseline.json"
STATE = ROOT / "experiments/search_state.json"
REGISTRY = ROOT / "experiments/registry.tsv"
CANDIDATES = (
    {"candidate": "spectral_stronger", "spectral_weight": 2.0, "aggregation": "mean"},
    {"candidate": "spectral_weaker", "spectral_weight": 0.5, "aggregation": "mean"},
    {"candidate": "spectral_robust", "spectral_weight": 1.0, "aggregation": "median"},
)


def decide(score: float, f1: float, baseline_score: float, baseline_f1: float) -> bool:
    """Require a strict primary-metric gain without an F1 regression."""
    return score > baseline_score and f1 >= baseline_f1


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "assess"))
    args = parser.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    baseline = json.loads(ACCEPTED.read_text(encoding="utf-8"))
    index = state["next_index"]
    if args.command == "prepare":
        if index >= len(CANDIDATES):
            print("candidate_available=false")
            return
        write_json(ACTIVE, CANDIDATES[index])
        print("candidate_available=true")
        print(f"candidate_name={CANDIDATES[index]['candidate']}")
        return

    if index >= len(CANDIDATES):
        raise ValueError("no candidate was prepared")
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    name = CANDIDATES[index]["candidate"]
    matches = [r for r in records if r["description"] == name and r["status"] == "candidate_result"]
    if not matches:
        raise ValueError(f"no completed result for {name}")
    latest = matches[-1]
    improved = decide(
        float(latest["balanced_accuracy"]), float(latest["f1"]),
        baseline["balanced_accuracy"], baseline["f1"],
    )
    state["next_index"] = index + 1
    write_json(STATE, state)
    if improved:
        write_json(ACCEPTED, {
            **CANDIDATES[index],
            "balanced_accuracy": float(latest["balanced_accuracy"]),
            "f1": float(latest["f1"]),
        })
    else:
        write_json(ACTIVE, {
            key: baseline[key] for key in ("candidate", "spectral_weight", "aggregation")
        })
    print(f"improved={'true' if improved else 'false'}")
    print(f"candidate_name={name}")
    print(f"experiment_id={latest['experiment_id']}")


if __name__ == "__main__":
    main()
