"""Audit TORGO recording conditions without publishing paths or speaker identities."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

DIRECTORY = re.compile(r"wav_(arrayMic|headMic)_([FM]C?\d{2})(?:S(\d{2}))?")
CLASS = {
    "F_Con": ("control", "female"),
    "F_Dys": ("dysarthria", "female"),
    "M_Con": ("control", "male"),
    "M_Dys": ("dysarthria", "male"),
}
SPLITS = ("train", "validation", "test")
LABELS = ("control", "dysarthria")


def audit(rows: list[dict[str, str]]) -> dict[str, object]:
    """Return aggregate coverage and reject inconsistent TORGO metadata."""
    if not rows:
        raise ValueError("manifest is empty")
    speakers: dict[str, tuple[str, str]] = {}
    coverage: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"recordings": 0, "speakers": set(), "microphones": defaultdict(int),
                 "sessions": set(), "unspecified_session_recordings": 0}
    )
    for row in rows:
        path = Path(row["audio_path"])
        directory = DIRECTORY.fullmatch(path.parent.name)
        if directory is None or path.parent.parent.name not in CLASS:
            raise ValueError("manifest contains a path outside the expected TORGO layout")
        microphone, folder_speaker, session = directory.groups()
        label, _ = CLASS[path.parent.parent.name]
        split = row.get("split")
        if split not in SPLITS or row["speaker_id"] != folder_speaker or row["label"] != label:
            raise ValueError("manifest split, label, or speaker conflicts with TORGO layout")
        speaker = row["speaker_id"]
        identity = (split, label)
        if speaker in speakers and speakers[speaker] != identity:
            raise ValueError("speaker appears in multiple splits or classes")
        speakers[speaker] = identity
        group = coverage[identity]
        group["recordings"] += 1
        group["speakers"].add(speaker)
        group["microphones"][microphone] += 1
        if session is None:
            group["unspecified_session_recordings"] += 1
        else:
            group["sessions"].add((speaker, session))

    by_split: dict[str, dict[str, dict[str, object]]] = {}
    for split in SPLITS:
        by_split[split] = {}
        for label in LABELS:
            group = coverage[(split, label)]
            by_split[split][label] = {
                "recordings": group["recordings"],
                "speakers": len(group["speakers"]),
                "microphones": {mic: group["microphones"][mic]
                                for mic in ("arrayMic", "headMic")},
                "speaker_sessions_with_id": len(group["sessions"]),
                "unspecified_session_recordings": group["unspecified_session_recordings"],
            }
    return {
        "recordings": len(rows),
        "speakers": len(speakers),
        "speaker_disjoint": True,
        "validation_speakers_per_class": {
            label: by_split["validation"][label]["speakers"] for label in LABELS
        },
        "by_split_and_label": by_split,
        "interpretation": (
            "Coverage audit only. It does not measure model accuracy, clinical severity, "
            "or therapy response. One validation speaker per class cannot establish "
            "new-speaker reliability."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/torgo_conditions.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("audit output must stay inside the Git-ignored artifacts directory")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        report = audit(list(csv.DictReader(handle)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"audited {report['recordings']} recordings across {report['speakers']} speakers")
    print("condition coverage saved to private artifacts directory")


if __name__ == "__main__":
    main()
