"""Build a candidate TORGO manifest from its published directory layout.

This script creates metadata only; it never copies or uploads audio.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

SPEAKER_RE = re.compile(r"wav_(?:arrayMic|headMic)_([FM]C?\d{2})(?:S\d{2})?$")
LABELS = {"F_Con": "control", "M_Con": "control", "F_Dys": "dysarthria", "M_Dys": "dysarthria"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="extracted TORGO archive directory")
    parser.add_argument("output", type=Path, help="private output CSV")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for audio in sorted(args.root.rglob("*.wav")):
        try:
            label = LABELS[audio.relative_to(args.root).parts[0]]
        except (IndexError, KeyError) as exc:
            raise SystemExit(f"unexpected class directory: {audio}") from exc
        match = SPEAKER_RE.fullmatch(audio.parent.name)
        if not match:
            raise SystemExit(f"cannot derive speaker from directory: {audio.parent}")
        rows.append({"audio_path": str(audio.resolve()), "speaker_id": match.group(1), "label": label,
                     "severity": "", "transcript": ""})

    if not rows:
        raise SystemExit("no WAV files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "speaker_id", "label", "severity", "transcript"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows for {len({r['speaker_id'] for r in rows})} speakers to {args.output}")


if __name__ == "__main__":
    main()
