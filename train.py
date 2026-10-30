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

    # A candidate must never inspect held-out test audio during iteration.
    active_rows = [row for row in rows if row["split"] in {"train", "validation"}]
    extracted = {row["audio_path"]: extract(row["audio_path"]) for row in active_rows}
    vectors = {path: pair[0] for path, pair in extracted.items()}
    baseline_vectors = {path: pair[1] for path, pair in extracted.items()}
    by_speaker: dict[str, list[np.ndarray]] = defaultdict(list)
    baseline_by_speaker: dict[str, list[np.ndarray]] = defaultdict(list)
    speaker_labels: dict[str, str] = {}
    for row in rows:
        if row["split"] == "train":
            speaker = row["speaker_id"]
            if speaker in speaker_labels and speaker_labels[speaker] != row["label"]:
                raise ValueError(f"conflicting labels for training speaker {speaker}")
            speaker_labels[speaker] = row["label"]
            by_speaker[speaker].append(vectors[row["audio_path"]])
            baseline_by_speaker[speaker].append(baseline_vectors[row["audio_path"]])
    training_features = {
        speaker: aggregate(recordings, axis=0) for speaker, recordings in by_speaker.items()
    }
    speaker_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "validation":
            speaker_rows[row["speaker_id"]].append(row)
    validation_features = {
        speaker: aggregate([vectors[row["audio_path"]] for row in recordings], axis=0)
        for speaker, recordings in speaker_rows.items()
    }
    all_labels = {**speaker_labels, **{
        speaker: recordings[0]["label"] for speaker, recordings in speaker_rows.items()
    }}
    predictions = predict_speakers(
        training_features, all_labels, validation_features, settings
    )
    write_predictions(predictions, arguments.output)
    output = Path(arguments.output)
    write_predictions(
        leave_one_speaker_out(training_features, speaker_labels, settings),
        output.with_name("cv_predictions.jsonl"),
    )
    if (baseline["aggregation"] == settings["aggregation"]
            and baseline_feature_set == feature_set):
        baseline_features = training_features
    else:
        baseline_aggregate = np.mean if baseline["aggregation"] == "mean" else np.median
        baseline_features = {
            speaker: baseline_aggregate(recordings, axis=0)
            for speaker, recordings in baseline_by_speaker.items()
        }
    write_predictions(
        leave_one_speaker_out(baseline_features, speaker_labels, baseline),
        output.with_name("baseline_cv_predictions.jsonl"),
    )
    print(f"audio baseline wrote {len(predictions)} speaker predictions")


if __name__ == "__main__":
    main()
