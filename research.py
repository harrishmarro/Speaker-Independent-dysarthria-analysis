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
ALLOWED_PROGRAMS = {"security.py", "prepare.py", "train.py", "evaluate.py"}


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
    predictions = output / "predictions.jsonl"
    metrics_path = output / "metrics.json"
    cv_metrics_path = output / "cv_metrics.json"
    baseline_cv_metrics_path = output / "baseline_cv_metrics.json"
    started = time.monotonic()

    run([sys.executable, "security.py", "train.py"], timeout=60)
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
    run(
        [sys.executable, "train.py", "--manifest", str(prepared), "--output", str(predictions)],
        timeout=1800,
    )
    run(
        [
            sys.executable,
            "evaluate.py",
            "--predictions",
            str(predictions),
            "--output",
            str(metrics_path),
        ],
        timeout=60,
    )
    for prediction_file, result_file in (
        ("cv_predictions.jsonl", cv_metrics_path),
        ("baseline_cv_predictions.jsonl", baseline_cv_metrics_path),
    ):
        run(
            [sys.executable, "evaluate.py", "--predictions", str(output / prediction_file),
             "--output", str(result_file)],
            timeout=60,
        )
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
            "cv_speakers": str(sum(
                cv_metrics[key] for key in (
                    "true_positive", "true_negative", "false_positive", "false_negative"
                )
            )),
        }
    )
    print(f"recorded {experiment_id} as {status}")


if __name__ == "__main__":
    main()
