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
MODEL_HISTORY = ROOT / "experiments/Model_used.tsv"
APPROVALS = ROOT / "autoresearch/approvals.json"
CANDIDATES = (
    {"candidate": "speaker_knn_1", "spectral_weight": 1.0, "aggregation": "mean",
     "classifier": "speaker_knn", "neighbors": 1},
    {"candidate": "speaker_knn_3", "spectral_weight": 1.0, "aggregation": "mean",
     "classifier": "speaker_knn", "neighbors": 3},
    {"candidate": "speaker_knn_5", "spectral_weight": 1.0, "aggregation": "mean",
     "classifier": "speaker_knn", "neighbors": 5},
    {"candidate": "ridge_0_1", "spectral_weight": 1.0, "aggregation": "mean",
     "classifier": "ridge", "ridge_alpha": 0.1},
    {"candidate": "ridge_1", "spectral_weight": 1.0, "aggregation": "mean",
     "classifier": "ridge", "ridge_alpha": 1.0},
    {"candidate": "ridge_10", "spectral_weight": 1.0, "aggregation": "mean",
     "classifier": "ridge", "ridge_alpha": 10.0},
)


def decide(
    score: float, f1: float, baseline_score: float, baseline_f1: float,
    cv_score: float, cv_f1: float, baseline_cv_score: float, baseline_cv_f1: float,
) -> bool:
    """Require gains on validation and training-speaker CV without F1 regression."""
    return (score > baseline_score and f1 >= baseline_f1
            and cv_score > baseline_cv_score and cv_f1 >= baseline_cv_f1)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "assess"))
    args = parser.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    baseline = json.loads(ACCEPTED.read_text(encoding="utf-8"))
    approved = set(json.loads(APPROVALS.read_text(encoding="utf-8"))["approved_approaches"])
    index = state["next_index"]
    if args.command == "prepare":
        if index >= len(CANDIDATES):
            print("candidate_available=false")
            print("review_required=true")
            print("review_reason=catalog_exhausted")
            return
        approach = CANDIDATES[index].get("classifier", "centroid")
        if approach not in approved:
            print("candidate_available=false")
            print("review_required=true")
            print("review_reason=approach_approval_required")
            print(f"candidate_name={CANDIDATES[index]['candidate']}")
            return
        write_json(ACTIVE, CANDIDATES[index])
        print("candidate_available=true")
        print(f"candidate_name={CANDIDATES[index]['candidate']}")
        return

    if index >= len(CANDIDATES):
        raise ValueError("no candidate was prepared")
    if CANDIDATES[index].get("classifier", "centroid") not in approved:
        raise ValueError("candidate approach has not been approved")
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
        float(latest["cv_balanced_accuracy"]), float(latest["cv_f1"]),
        float(latest["baseline_cv_balanced_accuracy"]),
        float(latest["baseline_cv_f1"]),
    )
    state["next_index"] = index + 1
    write_json(STATE, state)
    if improved:
        write_json(ACCEPTED, {
            **CANDIDATES[index],
            "balanced_accuracy": float(latest["balanced_accuracy"]),
            "f1": float(latest["f1"]),
            "cv_balanced_accuracy": float(latest["cv_balanced_accuracy"]),
            "cv_f1": float(latest["cv_f1"]),
        })
        with MODEL_HISTORY.open("a", encoding="utf-8") as handle:
            handle.write(
                "\t".join((
                    name,
                    latest.get("model_code_commit", "unavailable"),
                    latest["experiment_id"],
                    latest["balanced_accuracy"],
                    latest["f1"],
                    "proposed_for_review",
                    "Candidate exceeded the accepted validation baseline; requires PR review.",
                )) + "\n"
            )
    else:
        write_json(ACTIVE, {
            key: value for key, value in baseline.items()
            if key not in ("balanced_accuracy", "f1", "cv_balanced_accuracy", "cv_f1")
        })
    print(f"improved={'true' if improved else 'false'}")
    print(f"candidate_name={name}")
    print(f"experiment_id={latest['experiment_id']}")


if __name__ == "__main__":
    main()
