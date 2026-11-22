#!/usr/bin/env python3
"""Controlled experiment runner inspired by autoresearch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil

# The trusted controller uses an allowlisted, shell-free runner.
import subprocess  # nosec B404
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "experiments" / "registry.tsv"
ACTIVE_CANDIDATE = ROOT / "configs" / "research" / "active.json"
ACCEPTED_BASELINE = ROOT / "configs" / "research" / "baseline.json"
CV_CODE = ROOT / "src" / "dysarthria" / "acoustic_cv.py"
MFCC_CODE = ROOT / "src" / "dysarthria" / "mfcc_features.py"
RHYTHM_CODE = ROOT / "src" / "dysarthria" / "rhythm_features.py"
ALLOWED_PROGRAMS = {"security.py", "prepare.py", "train.py", "evaluate.py", "controls.py"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "unavailable"
    result = subprocess.run(  # nosec B603
        [git, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def run(command: list[str], *, timeout: int) -> None:
    if len(command) < 2 or command[0] != sys.executable or command[1] not in ALLOWED_PROGRAMS:
        raise ValueError("research controller rejected a non-allowlisted command")
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONHASHSEED": "0",
    }
    # The executable and script are allowlisted above; shell execution is disabled.
    subprocess.run(  # nosec B603
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=timeout,
    )


def write_candidate_manifest(prepared: Path, destination: Path) -> int:
    """Copy the prepared manifest without held-out test rows; return rows withheld."""
    with prepared.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    visible = [row for row in rows if row["split"] in {"train", "validation"}]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(visible)
    return len(rows) - len(visible)


def has_mic_provenance(manifest: Path) -> bool:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return "mic" in (csv.DictReader(handle).fieldnames or [])


def score_mic_swaps(prepared: Path, candidate_manifest: Path, output: Path) -> dict[str, str]:
    """Run the locked controls and score every mic-swap prediction file.

    Returns registry fields; empty when the manifest has no mic provenance, in which case
    ``autoresearch/decide.py`` refuses to assess the result.
    """
    from autoresearch.decide import CONTROL_NAMES, reported_variants, required_variants

    if not has_mic_provenance(prepared):
        return {}
    run([sys.executable, "controls.py", "--manifest", str(candidate_manifest),
         "--output-dir", str(output)], timeout=1800)
    feature_set = str(json.loads(ACTIVE_CANDIDATE.read_text()).get("feature_set", "spectral"))
    variants = [*required_variants(feature_set), *reported_variants(feature_set)]
    files = {f"swap_{variant}_{d}": f"swap_{variant}_{d}.jsonl"
             for variant in [*variants, "baseline"] for d in ("h2a", "a2h")}
    files.update({f"control_{name}_{d}": f"control_{name}_{d}.jsonl"
                  for name in CONTROL_NAMES for d in ("h2a", "a2h")})
    record = {"swap_variants": ",".join(required_variants(feature_set))}
    for field, name in files.items():
        if not (output / name).exists():
            raise FileNotFoundError(f"candidate did not write required {name}")
        result = output / f"{field}_metrics.json"
        run([sys.executable, "evaluate.py", "--predictions", str(output / name),
             "--manifest", str(prepared), "--split", "development", "--both-mics",
             "--output", str(result)], timeout=60)
        values = json.loads(result.read_text(encoding="utf-8"))
        record[f"{field}_ba"] = f"{values['balanced_accuracy']:.6f}"
        record[f"{field}_auc"] = f"{values['auc']:.6f}" if "auc" in values else ""
    for name in CONTROL_NAMES:
        result = output / f"control_{name}_cv_metrics.json"
        run([sys.executable, "evaluate.py", "--predictions",
             str(output / f"control_{name}_cv.jsonl"), "--manifest", str(prepared),
             "--split", "development", "--output", str(result)], timeout=60)
        values = json.loads(result.read_text(encoding="utf-8"))
        record[f"control_{name}_cv_auc"] = f"{values.get('auc', float('nan')):.6f}"
    return record


def existing_signatures() -> set[str]:
    if not REGISTRY.exists():
        return set()
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        return {row["signature"] for row in csv.DictReader(handle, delimiter="\t")}


def append_registry(record: dict[str, str]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    existed = REGISTRY.exists()
    if existed:
        with REGISTRY.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            previous_columns = list(reader.fieldnames or [])
            previous_rows = list(reader)
        columns = list(dict.fromkeys([*previous_columns, *record]))
        if columns != previous_columns:
            temporary = REGISTRY.with_suffix(".migration.tsv")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=columns, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(previous_rows)
            temporary.replace(REGISTRY)
    else:
        columns = list(record)
    with REGISTRY.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )
        if not existed:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "experiment"), default="smoke")
    parser.add_argument("--manifest", default="data/sample_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    manifest = ROOT / arguments.manifest
    signature_input = ":".join(
        [arguments.mode, str(arguments.seed), digest(ROOT / "train.py"), digest(CV_CODE),
         digest(MFCC_CODE), digest(RHYTHM_CODE), digest(ROOT / "controls.py"),
         digest(ACTIVE_CANDIDATE), digest(ACCEPTED_BASELINE), digest(manifest)]
    )
    signature = hashlib.sha256(signature_input.encode()).hexdigest()[:16]
    if signature in existing_signatures():
        print(f"experiment already recorded: {signature}")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"EXP-{timestamp}-{signature[:6]}"
    output = ROOT / "artifacts" / experiment_id
    prepared = output / "prepared_manifest.csv"
    candidate_manifest = output / "candidate_manifest.csv"
    predictions = output / "predictions.jsonl"
    metrics_path = output / "metrics.json"
    cv_metrics_path = output / "cv_metrics.json"
    baseline_cv_metrics_path = output / "baseline_cv_metrics.json"
    started = time.monotonic()

    run([sys.executable, "security.py", "train.py"], timeout=60)
    run([sys.executable, "security.py", "src/dysarthria/mfcc_features.py"], timeout=60)
    run([sys.executable, "security.py", "src/dysarthria/rhythm_features.py"], timeout=60)
    run(
        [
            sys.executable,
            "prepare.py",
            "--manifest",
            str(manifest),
            "--output",
            str(prepared),
            "--seed",
            str(arguments.seed),
        ],
        timeout=60,
    )
    # The candidate never receives held-out test rows; the locked manifest stays here.
    write_candidate_manifest(prepared, candidate_manifest)
    run(
        [sys.executable, "train.py", "--manifest", str(candidate_manifest),
         "--output", str(predictions)],
        timeout=1800,
    )
    for prediction_file, split, result_file in (
        ("predictions.jsonl", "validation", metrics_path),
        ("cv_predictions.jsonl", "development", cv_metrics_path),
        ("baseline_cv_predictions.jsonl", "development", baseline_cv_metrics_path),
    ):
        run(
            [sys.executable, "evaluate.py", "--predictions", str(output / prediction_file),
             "--manifest", str(prepared), "--split", split, "--output", str(result_file)],
            timeout=60,
        )
    swap_record = score_mic_swaps(prepared, candidate_manifest, output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cv_metrics = json.loads(cv_metrics_path.read_text(encoding="utf-8"))
    baseline_cv_metrics = json.loads(baseline_cv_metrics_path.read_text(encoding="utf-8"))
    duration = time.monotonic() - started
    status = "smoke_only" if arguments.mode == "smoke" else "candidate_result"
    append_registry(
        {
            "experiment_id": experiment_id,
            "timestamp_utc": timestamp,
            "signature": signature,
            "mode": arguments.mode,
            "seed": str(arguments.seed),
            "dataset_sha256": digest(manifest),
            "candidate_sha256": digest(ROOT / "train.py"),
            "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
            "f1": f"{metrics['f1']:.6f}",
            "recall": f"{metrics['recall']:.6f}",
            "runtime_seconds": f"{duration:.3f}",
            "status": status,
            "description": json.loads(ACTIVE_CANDIDATE.read_text())["candidate"],
            "source_commit": source_commit(),
            "model_code_commit": source_commit(),
            "candidate_config_sha256": digest(ACTIVE_CANDIDATE),
            "candidate_config": json.dumps(
                json.loads(ACTIVE_CANDIDATE.read_text()), sort_keys=True, separators=(",", ":")
            ),
            "cv_balanced_accuracy": f"{cv_metrics['balanced_accuracy']:.6f}",
            "cv_f1": f"{cv_metrics['f1']:.6f}",
            "baseline_cv_balanced_accuracy": f"{baseline_cv_metrics['balanced_accuracy']:.6f}",
            "baseline_cv_f1": f"{baseline_cv_metrics['f1']:.6f}",
            "cv_speakers": str(cv_metrics["speakers"]),
            "cv_pool": "train+validation",
            "cv_auc": f"{cv_metrics['auc']:.6f}" if "auc" in cv_metrics else "",
            "baseline_cv_auc": (
                f"{baseline_cv_metrics['auc']:.6f}" if "auc" in baseline_cv_metrics else ""
            ),
            **swap_record,
        }
    )
    print(f"recorded {experiment_id} as {status}")


if __name__ == "__main__":
    main()
