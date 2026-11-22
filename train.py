#!/usr/bin/env python3
"""Speaker-balanced acoustic candidate: spectral, MFCC or speech-rhythm features."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from dysarthria.acoustic_cv import MICROPHONES, leave_one_speaker_out, mic_swap, predict_speakers
from dysarthria.mfcc_features import mfcc_summary
from dysarthria.rhythm_features import PRIMARY_OFFSET_DB, rhythm_variants
from locked.candidate_io import read_prepared_manifest, write_predictions

SWAP_DIRECTIONS = {"h2a": ("headMic", "arrayMic"), "a2h": ("arrayMic", "headMic")}
MAIN = "main"


def spectral_vector(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Whole-recording spectral summary (unchanged from the accepted baseline)."""
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
    return np.concatenate((summary, np.asarray(spectral_bands, dtype=np.float32)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    rows = read_prepared_manifest(arguments.manifest)
    root = Path(__file__).resolve().parent
    settings = json.loads((root / "configs/research/active.json").read_text())
    baseline = json.loads((root / "configs/research/baseline.json").read_text())
    for config in (settings, baseline):
        if config["aggregation"] not in ("mean", "median"):
            raise ValueError("unsupported aggregation")
        if config.get("feature_set", "spectral") not in ("spectral", "mfcc", "rhythm"):
            raise ValueError("unsupported feature set")
    feature_set = settings.get("feature_set", "spectral")
    baseline_feature_set = baseline.get("feature_set", "spectral")

    def vectors(path: str, family: str) -> dict[str, np.ndarray]:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        signal = np.asarray(audio, dtype=np.float32)
        if signal.ndim == 2:
            signal = signal.mean(axis=1)
        if signal.size == 0 or sample_rate <= 0:
            raise ValueError(f"empty or invalid audio: {path}")
        if family == "rhythm":
            variants = rhythm_variants(signal.astype(np.float64), sample_rate, Path(path).name)
            return {MAIN: variants[f"vad{int(PRIMARY_OFFSET_DB)}"], **variants}
        signal = signal - float(signal.mean())
        scale = float(np.max(np.abs(signal)))
        if scale > 0:
            signal = signal / scale
        if family == "mfcc":
            return {MAIN: mfcc_summary(signal, sample_rate)}
        return {MAIN: spectral_vector(signal, sample_rate)}

    # A candidate must never inspect held-out test audio during iteration. The research
    # controller already strips test rows; refuse a manifest that still contains them.
    if any(row["split"] not in {"train", "validation"} for row in rows):
        raise ValueError("candidate manifest must contain only train and validation rows")
    labels: dict[str, str] = {}
    splits: dict[str, str] = {}
    has_mics = all(row.get("mic") in MICROPHONES for row in rows)
    # recordings[family][variant][(speaker, mic or "")] -> per-recording vectors
    recordings: dict[str, dict[str, dict[tuple[str, str], list[np.ndarray]]]] = {}
    families = {"candidate": feature_set, "baseline": baseline_feature_set}
    skipped = 0
    for row in rows:
        speaker = row["speaker_id"]
        if labels.setdefault(speaker, row["label"]) != row["label"]:
            raise ValueError(f"conflicting labels for speaker {speaker}")
        if splits.setdefault(speaker, row["split"]) != row["split"]:
            raise ValueError(f"speaker {speaker} appears in more than one split")
        cache: dict[str, dict[str, np.ndarray]] = {}
        try:
            for family in set(families.values()):
                cache[family] = vectors(row["audio_path"], family)
        except (ValueError, RuntimeError):
            skipped += 1  # unreadable, empty or sub-25 ms recording: skipped for every role
            continue
        for role, family in families.items():
            for variant, vector in cache[family].items():
                key = (speaker, row["mic"] if has_mics else "")
                recordings.setdefault(role, {}).setdefault(variant, defaultdict(list))[
                    key].append(vector)

    def aggregate(config: dict[str, object], items: list[np.ndarray]) -> np.ndarray:
        stacked = np.stack(items)
        return np.nanmean(stacked, axis=0) if config["aggregation"] == "mean" else (
            np.nanmedian(stacked, axis=0))

    def per_speaker(role: str, config: dict[str, object], variant: str) -> dict[str, np.ndarray]:
        pooled: dict[str, list[np.ndarray]] = defaultdict(list)
        for (speaker, _), items in recordings[role][variant].items():
            pooled[speaker].extend(items)
        return {speaker: aggregate(config, items) for speaker, items in pooled.items()}

    def per_speaker_mic(role: str, config: dict[str, object], variant: str
                        ) -> dict[tuple[str, str], np.ndarray]:
        return {key: aggregate(config, items) for key, items in recordings[role][variant].items()}

    candidate_features = per_speaker("candidate", settings, MAIN)
    baseline_features = per_speaker("baseline", baseline, MAIN)

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
    # Same-mic evidence: leave-one-speaker-out over every development speaker, scored
    # identically for the candidate and the accepted baseline in this same run.
    write_predictions(
        leave_one_speaker_out(candidate_features, labels, settings),
        output.with_name("cv_predictions.jsonl"),
    )
    write_predictions(
        leave_one_speaker_out(baseline_features, labels, baseline),
        output.with_name("baseline_cv_predictions.jsonl"),
    )
    # Cross-microphone evidence: every candidate variant and the baseline, both directions.
    if has_mics:
        jobs = [("candidate", settings, variant, variant) for variant in recordings["candidate"]]
        jobs.append(("baseline", baseline, MAIN, "baseline"))
        for role, config, variant, name in jobs:
            by_mic = per_speaker_mic(role, config, variant)
            for direction, (source, target) in SWAP_DIRECTIONS.items():
                write_predictions(
                    mic_swap(by_mic, labels, config, source, target),
                    output.with_name(f"swap_{name}_{direction}.jsonl"),
                )
    print(f"{feature_set} candidate wrote {len(predictions)} validation predictions; "
          f"skipped {skipped} unusable recordings")


if __name__ == "__main__":
    main()
