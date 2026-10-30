"""Deterministic utterance-level MFCC summaries for the classical baseline."""

from __future__ import annotations

import numpy as np


def mfcc_summary(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return mean and deviation of 13 MFCCs from sampled 25 ms frames.

    At most 160 frames are sampled uniformly over an utterance, matching the
    bounded whole-recording spectral extractor. No dataset-wide statistics are
    learned here; speaker-fold standardization occurs in acoustic_cv.py.
    """
    if sample_rate != 16000:
        raise ValueError("MFCC baseline requires 16 kHz audio")
    if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("MFCC baseline requires finite, nonempty mono audio")
    frame_length = min(400, signal.size)
    frame_count = min(160, max(1, signal.size // max(1, frame_length // 2)))
    starts = np.linspace(0, signal.size - frame_length, frame_count, dtype=int)
    frames = np.stack([signal[start : start + frame_length] for start in starts])
    power = np.abs(np.fft.rfft(frames * np.hanning(frame_length), n=512, axis=1)) ** 2

    mel_points = np.linspace(0.0, 2595.0 * np.log10(1.0 + 8000.0 / 700.0), 28)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    frequencies = np.fft.rfftfreq(512, 1.0 / sample_rate)
    filters = []
    for lower, center, upper in zip(hz_points[:-2], hz_points[1:-1], hz_points[2:],
                                    strict=True):
        rise = (frequencies - lower) / (center - lower)
        fall = (upper - frequencies) / (upper - center)
        triangle = np.maximum(0.0, np.minimum(rise, fall))
        filters.append(triangle / max(triangle.sum(), 1e-12))
    log_energies = np.log(np.maximum(power @ np.stack(filters, axis=1), 1e-12))
    basis = np.cos(
        np.pi / 26.0 * (np.arange(26)[:, None] + 0.5) * np.arange(13)[None, :]
    )
    coefficients = log_energies @ basis
    return np.concatenate((coefficients.mean(axis=0), coefficients.std(axis=0))).astype(
        np.float32
    )
