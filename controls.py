#!/usr/bin/env python3
"""Locked recording-condition controls, computed independently of the candidate.

For every development speaker this scores two nuisance-only models under the same
speaker-held-out microphone swap the candidate faces:

- ``channel``: noise floor and SNR only (ridge, alpha 1).
- ``silence``: mean cepstrum of non-speech frames only (ridge, alpha 10).

Their settings are fixed here, not in the candidate surface, so a candidate cannot weaken
the bar it has to clear.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from dysarthria.acoustic_cv import MICROPHONES, leave_one_speaker_out, mic_swap
from dysarthria.rhythm_features import control_vectors
from locked.candidate_io import read_prepared_manifest, write_predictions

CONTROLS: dict[str, dict[str, object]] = {
    "channel": {"feature_set": "channel", "classifier": "ridge", "ridge_alpha": 1.0,
                "spectral_weight": 1.0, "aggregation": "median"},
    "silence": {"feature_set": "silence", "classifier": "ridge", "ridge_alpha": 10.0,
                "spectral_weight": 1.0, "aggregation": "median"},
}
DIRECTIONS = {"h2a": ("headMic", "arrayMic"), "a2h": ("arrayMic", "headMic")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="candidate manifest (no test rows)")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    rows = read_prepared_manifest(arguments.manifest)
    if any(row["split"] not in {"train", "validation"} for row in rows):
        raise ValueError("controls must never see held-out test rows")
    if not all(row.get("mic") in MICROPHONES for row in rows):
        raise ValueError("controls need a mic column; rebuild the manifest with provenance")
    labels: dict[str, str] = {}
    by_key: dict[str, dict[tuple[str, str], list[np.ndarray]]] = {
        name: defaultdict(list) for name in CONTROLS
    }
    for row in rows:
        labels[row["speaker_id"]] = row["label"]
        try:
            audio, sample_rate = sf.read(row["audio_path"], dtype="float64", always_2d=False)
            signal = np.asarray(audio)
            vectors = control_vectors(signal.mean(axis=1) if signal.ndim == 2 else signal,
                                      sample_rate)
        except (ValueError, RuntimeError):
            continue  # unusable recording: too short or silent; skipped identically everywhere
        for name in CONTROLS:
            by_key[name][(row["speaker_id"], row["mic"])].append(vectors[name])

    output = Path(arguments.output_dir)
    for name, settings in CONTROLS.items():
        by_mic = {key: np.median(np.stack(items), axis=0) for key, items in by_key[name].items()}
        for direction, (source, target) in DIRECTIONS.items():
            write_predictions(mic_swap(by_mic, labels, settings, source, target),
                              output / f"control_{name}_{direction}.jsonl")
        pooled: dict[str, list[np.ndarray]] = defaultdict(list)
        for (speaker, _), items in by_key[name].items():
            pooled[speaker].extend(items)
        speakers = {speaker: np.median(np.stack(items), axis=0)
                    for speaker, items in pooled.items()}
        write_predictions(leave_one_speaker_out(speakers, labels, settings),
                          output / f"control_{name}_cv.jsonl")
    print(f"controls scored {len(labels)} development speakers")


if __name__ == "__main__":
    main()
