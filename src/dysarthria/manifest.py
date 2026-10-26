"""Dataset manifest validation and speaker-disjoint splitting."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

REQUIRED_COLUMNS = {"audio_path", "speaker_id", "label"}
ALLOWED_LABELS = {"control", "dysarthria"}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load and validate a CSV manifest without accessing the audio files."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError("manifest must contain at least one recording")

    for index, row in enumerate(rows, start=2):
        for column in REQUIRED_COLUMNS:
            if not row[column].strip():
                raise ValueError(f"row {index}: {column} must not be empty")
        if row["label"] not in ALLOWED_LABELS:
            raise ValueError(
                f"row {index}: label must be one of {sorted(ALLOWED_LABELS)}, "
                f"got {row['label']!r}"
            )
    _speaker_labels(rows)
    return rows


def _speaker_labels(rows: list[dict[str, str]]) -> dict[str, str]:
    labels_by_speaker: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_speaker[row["speaker_id"]].add(row["label"])
    ambiguous = sorted(speaker for speaker, labels in labels_by_speaker.items() if len(labels) != 1)
    if ambiguous:
        raise ValueError(f"speakers have conflicting labels: {', '.join(ambiguous)}")
    return {speaker: next(iter(labels)) for speaker, labels in labels_by_speaker.items()}


def assign_speaker_splits(
    rows: list[dict[str, str]],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Return rows with deterministic, label-stratified, speaker-wise split assignments."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("validation_ratio must be non-negative and leave room for test data")

    speakers_by_label: dict[str, list[str]] = defaultdict(list)
    for speaker, label in _speaker_labels(rows).items():
        speakers_by_label[label].append(speaker)

    split_by_speaker: dict[str, str] = {}
    rng = random.Random(seed)
    for label in sorted(speakers_by_label):
        speakers = sorted(speakers_by_label[label])
        rng.shuffle(speakers)
        total = len(speakers)
        train_end = max(1, round(total * train_ratio))
        train_end = min(train_end, total)
        validation_count = round(total * validation_ratio)
        if total >= 3 and validation_ratio > 0:
            validation_count = max(1, validation_count)
        validation_end = min(total, train_end + validation_count)
        for speaker in speakers[:train_end]:
            split_by_speaker[speaker] = "train"
        for speaker in speakers[train_end:validation_end]:
            split_by_speaker[speaker] = "validation"
        for speaker in speakers[validation_end:]:
            split_by_speaker[speaker] = "test"

    return [{**row, "split": split_by_speaker[row["speaker_id"]]} for row in rows]


def write_manifest(rows: list[dict[str, str]], path: str | Path) -> None:
    """Write rows using the union of their fields while preserving common field order."""
    preferred = ["audio_path", "speaker_id", "label", "severity", "transcript", "split"]
    available = {key for row in rows for key in row}
    columns = [column for column in preferred if column in available]
    columns.extend(sorted(available - set(columns)))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
