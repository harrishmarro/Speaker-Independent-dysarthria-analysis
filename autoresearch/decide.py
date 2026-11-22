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
    {"candidate": "mfcc_ridge_0_1", "feature_set": "mfcc", "spectral_weight": 1.0,
     "aggregation": "mean", "classifier": "ridge", "ridge_alpha": 0.1},
    {"candidate": "mfcc_ridge_1", "feature_set": "mfcc", "spectral_weight": 1.0,
     "aggregation": "mean", "classifier": "ridge", "ridge_alpha": 1.0},
    {"candidate": "mfcc_ridge_10", "feature_set": "mfcc", "spectral_weight": 1.0,
     "aggregation": "mean", "classifier": "ridge", "ridge_alpha": 10.0},
    # Prespecified rhythm candidate (plan P1): four features, median aggregation, no tuning.
    {"candidate": "rhythm_ridge_1", "feature_set": "rhythm", "spectral_weight": 1.0,
     "aggregation": "median", "classifier": "ridge", "ridge_alpha": 1.0},
)


def approach(candidate: dict[str, object]) -> str:
    """Gate a new feature family separately from its classifier head."""
    if candidate.get("feature_set") in ("mfcc", "rhythm"):
        return str(candidate["feature_set"])
    return str(candidate.get("classifier", "centroid"))


CV_POOL = "train+validation"
DIRECTIONS = ("h2a", "a2h")
CONTROL_NAMES = ("channel", "silence")
MIN_SWAP_BA = 0.70
MIN_MARGIN_OVER_CONTROLS = 0.15
MIN_SWAP_AUC = 0.75
MIN_AUC_MARGIN = 0.10  # ~4 of 42 case-control orderings with 6 controls and 7 cases
# Every listed variant must pass; a candidate cannot choose its favourable VAD setting.
REQUIRED_SWAP_VARIANTS = {"rhythm": ("vad10", "vad15", "vad20")}
REPORTED_SWAP_VARIANTS = {"rhythm": ("eq15",)}


def required_variants(feature_set: str) -> tuple[str, ...]:
    return REQUIRED_SWAP_VARIANTS.get(feature_set, ("main",))


def reported_variants(feature_set: str) -> tuple[str, ...]:
    return REPORTED_SWAP_VARIANTS.get(feature_set, ())


def swap_gate(
    candidate: dict[str, dict[str, float]],
    controls: dict[str, dict[str, float]],
    baseline: dict[str, float],
    *,
    candidate_auc: dict[str, dict[str, float]],
    control_auc: dict[str, dict[str, float]],
) -> tuple[bool, str]:
    """Promote only on cross-microphone evidence measured in one run.

    ``candidate``/``controls`` hold balanced accuracies and ``candidate_auc``/``control_auc``
    hold AUCs, keyed ``[variant or control][direction]``, from speaker-held-out mic swaps;
    ``baseline[direction]`` is the accepted baseline's balanced accuracy.

    Every required variant, in both directions, must:
    - reach ``MIN_SWAP_BA`` and beat the stronger control's BA by ``MIN_MARGIN_OVER_CONTROLS``;
    - reach ``MIN_SWAP_AUC`` and beat the stronger control's AUC by ``MIN_AUC_MARGIN``.
      Control AUC is direction-free, max(AUC, 1 - AUC): a nuisance cue whose ranking flips
      between microphones still carries label information. BA alone understates controls
      whose ranking transfers but whose threshold does not.
    Finally the worst-case BA must exceed the baseline's worst case.
    """
    for variant in candidate:
        for direction in DIRECTIONS:
            score = candidate[variant][direction]
            bar = max(controls[name][direction] for name in CONTROL_NAMES)
            if score < MIN_SWAP_BA:
                return False, f"{variant} {direction} BA {score:.3f} < {MIN_SWAP_BA}"
            if score < bar + MIN_MARGIN_OVER_CONTROLS:
                return False, (f"{variant} {direction} BA {score:.3f} within "
                               f"{MIN_MARGIN_OVER_CONTROLS} of control {bar:.3f}")
            auc = candidate_auc[variant][direction]
            auc_bar = max(max(control_auc[name][direction], 1 - control_auc[name][direction])
                          for name in CONTROL_NAMES)
            if auc < MIN_SWAP_AUC:
                return False, f"{variant} {direction} AUC {auc:.3f} < {MIN_SWAP_AUC}"
            if auc < auc_bar + MIN_AUC_MARGIN:
                return False, (f"{variant} {direction} AUC {auc:.3f} within "
                               f"{MIN_AUC_MARGIN} of control AUC {auc_bar:.3f}")
    worst = min(scores[d] for scores in candidate.values() for d in DIRECTIONS)
    baseline_worst = min(baseline[d] for d in DIRECTIONS)
    if worst <= baseline_worst:
        return False, f"worst-case swap BA {worst:.3f} <= baseline {baseline_worst:.3f}"
    return True, f"worst-case swap BA {worst:.3f} > baseline {baseline_worst:.3f}"


def gate_from_record(record: dict[str, str], feature_set: str) -> tuple[bool, str]:
    def score(key: str) -> float:
        value = record.get(key, "")
        if value == "":
            raise ValueError(f"result lacks {key}; rerun with a mic-provenance manifest")
        return float(value)

    variants = required_variants(feature_set)

    def table(prefix: str, names: tuple[str, ...], metric: str) -> dict[str, dict[str, float]]:
        return {name: {d: score(f"{prefix}_{name}_{d}_{metric}") for d in DIRECTIONS}
                for name in names}

    return swap_gate(
        table("swap", variants, "ba"),
        table("control", CONTROL_NAMES, "ba"),
        {d: score(f"swap_baseline_{d}_ba") for d in DIRECTIONS},
        candidate_auc=table("swap", variants, "auc"),
        control_auc=table("control", CONTROL_NAMES, "auc"),
    )


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
        candidate_approach = approach(CANDIDATES[index])
        if candidate_approach not in approved:
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
    if approach(CANDIDATES[index]) not in approved:
        raise ValueError("candidate approach has not been approved")
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    name = str(CANDIDATES[index]["candidate"])
    matches = [r for r in records if r["description"] == name and r["status"] == "candidate_result"]
    if not matches:
        raise ValueError(f"no completed result for {name}")
    latest = matches[-1]
    if latest.get("cv_pool") != CV_POOL:
        raise ValueError(f"{name} result predates same-run {CV_POOL} CV; rerun it")
    improved, reason = gate_from_record(
        latest, str(CANDIDATES[index].get("feature_set", "spectral"))
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
                    f"Passed the mic-swap gate ({reason}); requires PR review.",
                )) + "\n"
            )
    else:
        write_json(ACTIVE, {
            key: value for key, value in baseline.items()
            if key not in ("balanced_accuracy", "f1", "cv_balanced_accuracy", "cv_f1")
        })
    print(f"improved={'true' if improved else 'false'}")
    print(f"gate_reason={reason}")
    print(f"candidate_name={name}")
    print(f"experiment_id={latest['experiment_id']}")


if __name__ == "__main__":
    main()
