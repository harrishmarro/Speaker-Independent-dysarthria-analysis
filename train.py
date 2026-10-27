#!/usr/bin/env python3
"""Speaker-balanced, training-standardized acoustic baseline."""

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
        by_label[speaker_labels[speaker]].append(np.mean(recordings, axis=0))
    if set(by_label) != {"control", "dysarthria"}:
        raise ValueError("training split must contain both labels")
    training_vectors = np.stack([vector for values in by_label.values() for vector in values])
    feature_mean = training_vectors.mean(axis=0)
    feature_scale = training_vectors.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    centroids = {
        label: np.mean([(vector - feature_mean) / feature_scale for vector in values], axis=0)
        for label, values in by_label.items()
    }
    speaker_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "validation":
            speaker_rows[row["speaker_id"]].append(row)

    predictions = []
    for speaker_id, recordings in sorted(speaker_rows.items()):
        vector = np.mean([vectors[row["audio_path"]] for row in recordings], axis=0)
        vector = (vector - feature_mean) / feature_scale
        distances = {
            label: float(np.linalg.norm(vector - center))
            for label, center in centroids.items()
        }
        margin = distances["control"] - distances["dysarthria"]
        probability = float(1.0 / (1.0 + np.exp(-margin)))
        predictions.append({
            "speaker_id": speaker_id,
            "target": int(recordings[0]["label"] == "dysarthria"),
            "prediction": int(margin > 0),
            "probability": probability,
        })
    write_predictions(predictions, arguments.output)
    print(f"audio baseline wrote {len(predictions)} speaker predictions")


if __name__ == "__main__":
    main()
