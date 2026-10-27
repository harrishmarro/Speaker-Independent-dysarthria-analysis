#!/usr/bin/env python3
"""Speaker-balanced acoustic candidate with whole-recording spectral features."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

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
    if settings["spectral_weight"] not in (0.5, 1.0, 2.0):
        raise ValueError("unsupported spectral weight")
    if settings["aggregation"] not in ("mean", "median"):
        raise ValueError("unsupported aggregation")
    classifier = settings.get("classifier", "centroid")
    if classifier not in ("centroid", "speaker_knn"):
        raise ValueError("unsupported classifier")
    neighbors = settings.get("neighbors", 1)
    if neighbors not in (1, 3, 5):
        raise ValueError("unsupported neighbor count")
    aggregate = np.mean if settings["aggregation"] == "mean" else np.median

    def extract(path: str) -> np.ndarray:
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
        return np.concatenate((summary, bands))

    # A candidate must never inspect held-out test audio during iteration.
    active_rows = [row for row in rows if row["split"] in {"train", "validation"}]
    vectors = {row["audio_path"]: extract(row["audio_path"]) for row in active_rows}
    by_speaker: dict[str, list[np.ndarray]] = defaultdict(list)
    speaker_labels: dict[str, str] = {}
    for row in rows:
        if row["split"] == "train":
            speaker = row["speaker_id"]
            if speaker in speaker_labels and speaker_labels[speaker] != row["label"]:
                raise ValueError(f"conflicting labels for training speaker {speaker}")
            speaker_labels[speaker] = row["label"]
            by_speaker[speaker].append(vectors[row["audio_path"]])
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for speaker, recordings in by_speaker.items():
        by_label[speaker_labels[speaker]].append(aggregate(recordings, axis=0))
    if set(by_label) != {"control", "dysarthria"}:
        raise ValueError("training split must contain both labels")
    training_vectors = np.stack([vector for values in by_label.values() for vector in values])
    feature_mean = training_vectors.mean(axis=0)
    feature_scale = training_vectors.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    weights = np.ones(training_vectors.shape[1], dtype=np.float32)
    weights[8:] = settings["spectral_weight"]
    centroids = {
        label: np.mean(
            [((vector - feature_mean) / feature_scale) * weights for vector in values], axis=0
        )
        for label, values in by_label.items()
    }
    training_speakers = [
        (
            speaker,
            speaker_labels[speaker],
            ((aggregate(recordings, axis=0) - feature_mean) / feature_scale) * weights,
        )
        for speaker, recordings in sorted(by_speaker.items())
    ]
    speaker_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "validation":
            speaker_rows[row["speaker_id"]].append(row)

    predictions = []
    for speaker_id, recordings in sorted(speaker_rows.items()):
        vector = aggregate([vectors[row["audio_path"]] for row in recordings], axis=0)
        vector = ((vector - feature_mean) / feature_scale) * weights
        if classifier == "speaker_knn":
            ranked = sorted(
                (
                    (float(np.linalg.norm(vector - training_vector)), speaker, label)
                    for speaker, label, training_vector in training_speakers
                )
            )
            closest = ranked[:neighbors]
            probability = sum(label == "dysarthria" for _, _, label in closest) / len(closest)
        else:
            distances = {
                label: float(np.linalg.norm(vector - center))
                for label, center in centroids.items()
            }
            margin = distances["control"] - distances["dysarthria"]
            probability = float(1.0 / (1.0 + np.exp(-margin)))
        predictions.append({
            "speaker_id": speaker_id,
            "target": int(recordings[0]["label"] == "dysarthria"),
            "prediction": int(probability > 0.5),
            "probability": probability,
        })
    write_predictions(predictions, arguments.output)
    print(f"audio baseline wrote {len(predictions)} speaker predictions")


if __name__ == "__main__":
    main()
