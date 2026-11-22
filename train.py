#!/usr/bin/env python3
"""Speaker-balanced acoustic candidate with whole-recording spectral features."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from dysarthria.acoustic_cv import leave_one_speaker_out, predict_speakers
from dysarthria.mfcc_features import mfcc_summary
from locked.candidate_io import read_prepared_manifest, write_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    rows = read_prepared_manifest(arguments.manifest)
    settings = json.loads(
        (Path(__file__).resolve().parent / "configs/research/active.json").read_text()
    )
    baseline = json.loads(
        (Path(__file__).resolve().parent / "configs/research/baseline.json").read_text()
    )
    if settings["aggregation"] not in ("mean", "median"):
        raise ValueError("unsupported aggregation")
    feature_set = settings.get("feature_set", "spectral")
    baseline_feature_set = baseline.get("feature_set", "spectral")
    if feature_set not in ("spectral", "mfcc") or baseline_feature_set not in (
        "spectral", "mfcc"
    ):
        raise ValueError("unsupported feature set")
    aggregate = np.mean if settings["aggregation"] == "mean" else np.median

    def extract(path: str) -> tuple[np.ndarray, np.ndarray]:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        signal = np.asarray(audio, dtype=np.float32)
        if signal.ndim == 2:
            signal = signal.mean(axis=1)
        if signal.size == 0 or sample_rate <= 0:
            raise ValueError(f"empty or invalid audio: {path}")
        signal = signal - float(signal.mean())
        scale = float(np.max(np.abs(signal)))
        if scale > 0:
            signal = signal / scale
        frame_length = min(400, signal.size)
        frame_count = min(160, max(1, signal.size // max(1, frame_length // 2)))
        starts = np.linspace(0, signal.size - frame_length, frame_count, dtype=int)
        frames = np.stack([signal[start : start + frame_length] for start in starts])
        window = np.hanning(frame_length).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-6
        frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
        centroid = (spectrum * frequencies).sum(axis=1) / spectrum.sum(axis=1)
        energy = np.log1p((frames * frames).mean(axis=1) * 1_000.0)
        crossings = (
            np.mean(frames[:, 1:] * frames[:, :-1] < 0, axis=1)
            if frame_length > 1 else np.zeros(frame_count)
        )
        spectral_bands = []
        for lower, upper in zip((0, 300, 600, 1200, 2400, 4000),
                                (300, 600, 1200, 2400, 4000, 8000), strict=True):
            band = spectrum[:, (frequencies >= lower) & (frequencies < upper)]
            power = np.log1p(np.mean(band * band, axis=1)) if band.size else np.zeros(frame_count)
            spectral_bands.extend((power.mean(), power.std()))
        summary = np.array(
            [signal.std(), np.mean(np.abs(signal)), centroid.mean(), centroid.std(),
             energy.mean(), energy.std(), crossings.mean(), crossings.std()],
            dtype=np.float32,
        )
        bands = np.asarray(spectral_bands, dtype=np.float32)
        spectral = np.concatenate((summary, bands))
        candidate_vector = (
            mfcc_summary(signal, sample_rate) if feature_set == "mfcc" else spectral
        )
        baseline_vector = candidate_vector if baseline_feature_set == feature_set else (
            mfcc_summary(signal, sample_rate) if baseline_feature_set == "mfcc" else spectral
        )
        return candidate_vector, baseline_vector

    # A candidate must never inspect held-out test audio during iteration. The research
    # controller already strips test rows; refuse a manifest that still contains them.
    if any(row["split"] not in {"train", "validation"} for row in rows):
        raise ValueError("candidate manifest must contain only train and validation rows")
    extracted = {row["audio_path"]: extract(row["audio_path"]) for row in rows}
    labels: dict[str, str] = {}
    splits: dict[str, str] = {}
    candidate_recordings: dict[str, list[np.ndarray]] = defaultdict(list)
    baseline_recordings: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        speaker = row["speaker_id"]
        if labels.setdefault(speaker, row["label"]) != row["label"]:
            raise ValueError(f"conflicting labels for speaker {speaker}")
        if splits.setdefault(speaker, row["split"]) != row["split"]:
            raise ValueError(f"speaker {speaker} appears in more than one split")
        candidate_recordings[speaker].append(extracted[row["audio_path"]][0])
        baseline_recordings[speaker].append(extracted[row["audio_path"]][1])
    baseline_aggregate = np.mean if baseline["aggregation"] == "mean" else np.median
    candidate_features = {
        speaker: aggregate(items, axis=0) for speaker, items in candidate_recordings.items()
    }
    baseline_features = {
        speaker: baseline_aggregate(items, axis=0)
        for speaker, items in baseline_recordings.items()
    }

    def split_of(features: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
        return {speaker: vector for speaker, vector in features.items()
                if splits[speaker] == split}

    # Held-out validation report: fit on training speakers, predict validation speakers.
    output = Path(arguments.output)
    predictions = predict_speakers(
        split_of(candidate_features, "train"), labels,
        split_of(candidate_features, "validation"), settings,
    )
    write_predictions(predictions, output)
    # Selection evidence: leave-one-speaker-out over every development speaker, scored
    # identically for the candidate and the accepted baseline in this same run.
    write_predictions(
        leave_one_speaker_out(candidate_features, labels, settings),
        output.with_name("cv_predictions.jsonl"),
    )
    write_predictions(
        leave_one_speaker_out(baseline_features, labels, baseline),
        output.with_name("baseline_cv_predictions.jsonl"),
    )
    print(f"audio baseline wrote {len(predictions)} speaker predictions")


if __name__ == "__main__":
    main()
