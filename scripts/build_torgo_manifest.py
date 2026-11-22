"""Build a candidate TORGO manifest from its published directory layout.

This script creates metadata only; it never copies or uploads audio.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

SPEAKER_RE = re.compile(r"wav_(?:arrayMic|headMic)_([FM]C?\d{2})(?:S\d{2})?$")
# wav_<mic>_<speaker><session?>_<take>.wav; the same take is often recorded on both mics.
FILE_RE = re.compile(r"wav_(arrayMic|headMic)_([FM]C?\d{2})(S\d{2})?_(\d+)\.wav$")
LABELS = {"F_Con": "control", "M_Con": "control", "F_Dys": "dysarthria", "M_Dys": "dysarthria"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="extracted TORGO archive directory")
    parser.add_argument("output", type=Path, help="private output CSV")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    skipped: list[str] = []
    for audio in sorted(args.root.rglob("*.wav")):
        try:
            label = LABELS[audio.relative_to(args.root).parts[0]]
        except (IndexError, KeyError) as exc:
            raise SystemExit(f"unexpected class directory: {audio}") from exc
        match = SPEAKER_RE.fullmatch(audio.parent.name)
        if not match:
            raise SystemExit(f"cannot derive speaker from directory: {audio.parent}")
        provenance = FILE_RE.fullmatch(audio.name)
        if not provenance:
            # Stray files such as ".wav" or "*_0094.wavTMP.wav" are not numbered takes.
            skipped.append(audio.name)
            continue
        if provenance.group(2) != match.group(1):
            raise SystemExit(f"file speaker does not match its directory: {audio}")
        mic, speaker, session, take = provenance.groups()
        session = session or "S00"  # some speakers have a single, unnumbered session
        rows.append(
            {
                "audio_path": str(audio.resolve()),
                "speaker_id": speaker,
                "label": label,
                "severity": "",
                "transcript": "",
                "mic": mic,
                "session": session,
                "take_id": f"{speaker}{session}_{take}",
            }
        )

    if not rows:
        raise SystemExit("no WAV files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_path", "speaker_id", "label", "severity", "transcript",
                        "mic", "session", "take_id"],
        )
        writer.writeheader()
        writer.writerows(rows)
    speaker_count = len({row["speaker_id"] for row in rows})
    print(f"wrote {len(rows)} rows for {speaker_count} speakers to {args.output}")
    if skipped:
        print(f"skipped {len(skipped)} files without a take number: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
