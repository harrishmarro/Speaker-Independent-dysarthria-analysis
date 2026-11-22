#!/usr/bin/env python3
"""Locked frozen cross-corpus transfer (plan step P3).

Fit on the SOURCE corpus's development speakers (train + validation; its test speakers stay
locked), freeze scaling, model and decision threshold, then score every TARGET-corpus
speaker once. Ground truth comes from the target manifest, never from predictions.

The same frozen pipeline is run for the recording-condition controls, and each control is
also fitted *inside* the target corpus with leave-one-speaker-out. That shows how much the
target's own recording conditions reveal the label, which matters for corpora with a
single recording setup, where no microphone swap is possible.

Pass rule (prespecified): candidate AUC >= MIN_AUC, and for every control score set the
95% lower bound of the paired, class-stratified speaker-bootstrap AUC gain is above zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dysarthria.acoustic_cv import leave_one_speaker_out, predict_speakers
from dysarthria.audio_io import load_mono_16k
from dysarthria.metrics import roc_auc
from dysarthria.mfcc_features import mfcc_summary
from dysarthria.rhythm_features import PRIMARY_OFFSET_DB, control_vectors, rhythm_variants
from evaluate import evaluate

ROOT = Path(__file__).resolve().parent
MIN_AUC = 0.70
BOOTSTRAPS = 2000
SEED = 0
FROZEN: dict[str, dict[str, object]] = {
    "rhythm": {"feature_set": "rhythm", "classifier": "ridge", "ridge_alpha": 1.0,
               "spectral_weight": 1.0, "aggregation": "median"},
    "mfcc": {"feature_set": "mfcc", "classifier": "ridge", "ridge_alpha": 10.0,
             "spectral_weight": 1.0, "aggregation": "median"},
    "channel": {"feature_set": "channel", "classifier": "ridge", "ridge_alpha": 1.0,
                "spectral_weight": 1.0, "aggregation": "median"},
    "silence": {"feature_set": "silence", "classifier": "ridge", "ridge_alpha": 10.0,
                "spectral_weight": 1.0, "aggregation": "median"},
}
CANDIDATES = ("rhythm", "mfcc")
CONTROLS = ("channel", "silence")


def read_rows(path: Path, *, source: bool) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{path} is empty")
    if source:
        if any("split" not in row for row in rows):
            raise ValueError("source manifest must be a prepared manifest with a split column")
        rows = [row for row in rows if row["split"] in {"train", "validation"}]
    return rows


def recording_features(path: str, key: str) -> dict[str, np.ndarray]:
    signal = load_mono_16k(path)
    features = {"rhythm": rhythm_variants(signal, 16000, key)[f"vad{int(PRIMARY_OFFSET_DB)}"],
                "mfcc": mfcc_summary(signal / max(float(np.max(np.abs(signal))), 1e-12), 16000)}
    features.update(control_vectors(signal, 16000))
    return features


def speaker_features(rows: list[dict[str, str]], prefix: str
                     ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str], int]:
    """Median of per-recording features per speaker, for every feature family."""
    pooled: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, str] = {}
    skipped = 0
    for row in rows:
        speaker = f"{prefix}:{row['speaker_id']}"
        if labels.setdefault(speaker, row["label"]) != row["label"]:
            raise ValueError(f"conflicting labels for {speaker}")
        try:
            vectors = recording_features(row["audio_path"], Path(row["audio_path"]).name)
        except (ValueError, RuntimeError):
            skipped += 1
            continue
        for family, vector in vectors.items():
            pooled[family][speaker].append(vector)
    features = {family: {speaker: np.nanmedian(np.stack(items), axis=0)
                         for speaker, items in by_speaker.items()}
                for family, by_speaker in pooled.items()}
    return features, labels, skipped


def scores_of(predictions: list[dict[str, object]]) -> dict[str, float]:
    """Signed margin per speaker (logit of the probability) for paired AUC comparison."""
    out = {}
    for record in predictions:
        p = min(max(float(record["probability"]), 1e-9), 1 - 1e-9)  # type: ignore[arg-type]
        out[str(record["speaker_id"])] = float(np.log(p / (1 - p)))
    return out


def paired_gain_bound(candidate: dict[str, float], control: dict[str, float],
                      labels: dict[str, int]) -> tuple[float, float]:
    """Observed AUC gain over a direction-free control and its bootstrap 2.5th percentile.

    The control is oriented to its own better direction on the observed data, so a cue
    that ranks the classes backwards still counts as label information.
    """
    speakers = sorted(labels)
    y = np.array([labels[s] for s in speakers])
    a = np.array([candidate[s] for s in speakers])
    c = np.array([control[s] for s in speakers])
    if roc_auc(list(y), list(c)) < 0.5:
        c = -c
    observed = roc_auc(list(y), list(a)) - roc_auc(list(y), list(c))
    rng = np.random.default_rng(SEED)
    positives, negatives = np.where(y == 1)[0], np.where(y == 0)[0]
    gains = []
    for _ in range(BOOTSTRAPS):
        index = np.concatenate((rng.choice(positives, positives.size),
                                rng.choice(negatives, negatives.size)))
        gains.append(roc_auc(list(y[index]), list(a[index]))
                     - roc_auc(list(y[index]), list(c[index])))
    return observed, float(np.percentile(gains, 2.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, help="prepared source manifest")
    parser.add_argument("--target-manifest", required=True, help="external corpus manifest")
    parser.add_argument("--candidate", required=True, choices=CANDIDATES)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    source_rows = read_rows(Path(arguments.source_manifest), source=True)
    target_rows = read_rows(Path(arguments.target_manifest), source=False)
    source, source_labels, source_skipped = speaker_features(source_rows, "src")
    target, target_labels, target_skipped = speaker_features(target_rows, "tgt")
    labels = {**source_labels, **target_labels}
    truth = {speaker.split(":", 1)[1]: int(label == "dysarthria")
             for speaker, label in target_labels.items()}
    if set(truth.values()) != {0, 1}:
        raise ValueError("target corpus needs speakers from both classes")

    def strip(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{**record, "speaker_id": str(record["speaker_id"]).split(":", 1)[1]}
                for record in predictions]

    results: dict[str, dict[str, float | int]] = {}
    margins: dict[str, dict[str, float]] = {}
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for family in (arguments.candidate, *CONTROLS):
        frozen = strip(predict_speakers(source[family], labels, target[family], FROZEN[family]))
        results[f"{family}_transfer"] = evaluate(frozen, truth)
        margins[f"{family}_transfer"] = scores_of(frozen)
        within = strip(leave_one_speaker_out(target[family], labels, FROZEN[family]))
        results[f"{family}_within_target"] = evaluate(within, truth)
        margins[f"{family}_within_target"] = scores_of(within)
        (output / f"{family}_transfer.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in frozen), encoding="utf-8")

    candidate = margins[f"{arguments.candidate}_transfer"]
    gains = {name: paired_gain_bound(candidate, margins[name], truth)
             for name in margins if name.split("_", 1)[0] in CONTROLS}
    auc = float(results[f"{arguments.candidate}_transfer"]["auc"])
    failures = [f"{name}: gain {gain:+.3f}, lower bound {bound:+.3f}"
                for name, (gain, bound) in gains.items() if bound <= 0]
    passed = auc >= MIN_AUC and not failures
    summary = {
        "candidate": arguments.candidate,
        "frozen_settings": FROZEN,
        "code_sha256": hashlib.sha256((ROOT / "transfer.py").read_bytes()).hexdigest(),
        "source_speakers": len(source_labels), "target_speakers": len(truth),
        "skipped_recordings": {"source": source_skipped, "target": target_skipped},
        "results": results,
        "auc_gain_over_controls": {name: {"gain": g, "lower_95": b}
                                   for name, (g, b) in gains.items()},
        "verdict": "pass" if passed else "fail",
        "reason": (f"AUC {auc:.3f} >= {MIN_AUC} and gain over every control is positive"
                   if passed else (f"AUC {auc:.3f} < {MIN_AUC}; " if auc < MIN_AUC else "")
                   + "; ".join(failures)),
    }
    (output / "transfer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": summary["verdict"], "reason": summary["reason"],
                      "candidate_auc": auc}, sort_keys=True))


if __name__ == "__main__":
    main()
