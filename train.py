#!/usr/bin/env python3
"""Deterministic audio baseline for the autoresearch candidate surface."""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import soundfile as sf

from locked.candidate_io import read_prepared_manifest, write_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    rows = read_prepared_manifest(arguments.manifest)

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
        hop = max(1, frame_length // 2)
        starts = np.arange(0, max(1, signal.size - frame_length + 1), hop)[:160]
        frames = np.stack([signal[start : start + frame_length] for start in starts])
        window = np.hanning(frame_length).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-6
        frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
        centroid = (spectrum * frequencies).sum(axis=1) / spectrum.sum(axis=1)
        energy = np.log1p((frames * frames).mean(axis=1) * 1_000.0)
        crossings = np.mean(frames[:, 1:] * frames[:, :-1] < 0, axis=1)
        return np.array(
            [signal.std(), np.mean(np.abs(signal)), centroid.mean(), centroid.std(),
             energy.mean(), energy.std(), crossings.mean(), crossings.std()],
            dtype=np.float32,
        )

    vectors = {row["audio_path"]: extract(row["audio_path"]) for row in rows}
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_label[row["label"]].append(vectors[row["audio_path"]])
    if set(by_label) != {"control", "dysarthria"}:
        raise ValueError("training split must contain both labels")
    centroids = {label: np.mean(values, axis=0) for label, values in by_label.items()}
    speaker_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "validation":
            speaker_rows[row["speaker_id"]].append(row)

    predictions = []
    for speaker_id, recordings in sorted(speaker_rows.items()):
        vector = np.mean([vectors[row["audio_path"]] for row in recordings], axis=0)
        distances = {
            label: float(np.linalg.norm(vector - center))
            for label, center in centroids.items()
        }
        predicted = min(distances, key=distances.get)
        margin = distances["control"] - distances["dysarthria"]
        probability = float(1.0 / (1.0 + np.exp(-margin)))
        predictions.append({
            "speaker_id": speaker_id,
            "target": int(recordings[0]["label"] == "dysarthria"),
            "prediction": int(predicted == "dysarthria"),
            "probability": probability,
        })
    write_predictions(predictions, arguments.output)
    print(f"audio baseline wrote {len(predictions)} speaker predictions")


if __name__ == "__main__":
    main()
