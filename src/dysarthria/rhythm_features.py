"""Speech-rhythm features and recording-condition control features.

Everything here is deterministic, numpy-only and fitted to nothing: no dataset-wide
statistics are learned. Speaker-fold standardization happens in ``acoustic_cv.py``.

Speech activity is defined relative to each recording's own speech level (the 95th
percentile frame energy) rather than its noise floor, so a quieter or noisier session does
not by itself change which frames count as speech. ``VAD_OFFSETS_DB`` are prespecified;
the primary offset is 15 dB and the others exist only as a sensitivity check.
"""

from __future__ import annotations

import hashlib

import numpy as np

SAMPLE_RATE = 16000
FRAME = 400  # 25 ms
HOP = 160  # 10 ms
MAX_FRAMES = 1500
VAD_OFFSETS_DB = (10.0, 15.0, 20.0)
PRIMARY_OFFSET_DB = 15.0
EQUALIZED_FLOOR_DB = -30.0
MIN_PAUSE_FRAMES = 10  # 100 ms
RHYTHM_NAMES = ("occupancy", "pauses_per_second", "mean_run_seconds", "modulation_2_8hz")
CHANNEL_NAMES = ("noise_floor_db", "snr_db")
SILENCE_COEFFICIENTS = 6

_MEL = np.linspace(0.0, 2595.0 * np.log10(1.0 + 8000.0 / 700.0), 28)
_HZ = 700.0 * (10.0 ** (_MEL / 2595.0) - 1.0)
_FREQUENCIES = np.fft.rfftfreq(512, 1.0 / SAMPLE_RATE)
_FILTERS = []
for _lower, _center, _upper in zip(_HZ[:-2], _HZ[1:-1], _HZ[2:], strict=True):
    _triangle = np.maximum(0.0, np.minimum((_FREQUENCIES - _lower) / (_center - _lower),
                                           (_upper - _FREQUENCIES) / (_upper - _center)))
    _FILTERS.append(_triangle / max(float(_triangle.sum()), 1e-12))
_FILTERBANK = np.stack(_FILTERS, axis=1)
_DCT = np.cos(np.pi / 26.0 * (np.arange(26)[:, None] + 0.5) * np.arange(13)[None, :])
_WINDOW = np.hanning(FRAME)


def prepare_signal(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate, remove DC and peak-normalize, matching the spectral extractor."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError("rhythm features require 16 kHz audio")
    if signal.ndim != 1 or signal.size < FRAME or not np.all(np.isfinite(signal)):
        raise ValueError("rhythm features require finite mono audio of at least 25 ms")
    centered = signal.astype(np.float64) - float(signal.mean())
    peak = float(np.max(np.abs(centered)))
    if peak <= 0:
        raise ValueError("silent recording")
    return centered / peak


def equalize(signal: np.ndarray, key: str, floor_db: float = EQUALIZED_FLOOR_DB) -> np.ndarray:
    """Add seeded white noise so every recording reaches a common noise floor.

    Used only as a reported stress condition; the seed derives from ``key`` so the
    transform is reproducible and label-independent.
    """
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    noise = np.random.default_rng(seed).standard_normal(signal.size)
    return signal + noise * 10.0 ** (floor_db / 20.0)


def _frames(signal: np.ndarray) -> np.ndarray:
    starts = np.arange(0, signal.size - FRAME + 1, HOP)[:MAX_FRAMES]
    return np.stack([signal[start:start + FRAME] for start in starts])


def frame_energy_db(signal: np.ndarray) -> np.ndarray:
    frames = _frames(signal)
    return np.asarray(20.0 * np.log10(np.sqrt((frames * frames).mean(axis=1)) + 1e-9))


def speech_mask(energy_db: np.ndarray, offset_db: float) -> np.ndarray:
    return np.asarray(energy_db > np.percentile(energy_db, 95) - offset_db)


def rhythm_vector(energy_db: np.ndarray, offset_db: float) -> np.ndarray:
    """Occupancy, pause rate, mean speech-run length and 2-8 Hz envelope modulation.

    Measured between the first and last speech frame so leading and trailing silence,
    which mostly reflect how a session was cut, do not contribute.
    """
    mask = speech_mask(energy_db, offset_db)
    if not mask.any():
        return np.full(len(RHYTHM_NAMES), np.nan)
    first = int(np.argmax(mask))
    last = len(mask) - 1 - int(np.argmax(mask[::-1]))
    region = mask[first:last + 1]
    edges = np.diff(np.concatenate(([0], region.astype(int), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    runs = ends - starts
    gaps = starts[1:] - ends[:-1]
    seconds = max(region.size * HOP / SAMPLE_RATE, 0.1)
    envelope = energy_db[first:last + 1] - energy_db[first:last + 1].mean()
    modulation = np.nan
    if envelope.size >= 32:
        spectrum = np.abs(np.fft.rfft(envelope * np.hanning(envelope.size))) ** 2
        rates = np.fft.rfftfreq(envelope.size, HOP / SAMPLE_RATE)
        total = float(spectrum[rates > 0.5].sum())
        modulation = float(spectrum[(rates >= 2) & (rates <= 8)].sum()) / max(total, 1e-12)
    return np.array([
        float(region.mean()),
        float((gaps >= MIN_PAUSE_FRAMES).sum()) / seconds,
        float(runs.mean()) * HOP / SAMPLE_RATE,
        modulation,
    ])


def rhythm_variants(signal: np.ndarray, sample_rate: int, key: str) -> dict[str, np.ndarray]:
    """Rhythm vectors at every prespecified VAD offset plus the equalized stress condition."""
    prepared = prepare_signal(signal, sample_rate)
    energy = frame_energy_db(prepared)
    variants = {f"vad{int(offset)}": rhythm_vector(energy, offset) for offset in VAD_OFFSETS_DB}
    equalized = frame_energy_db(equalize(prepared, key))
    variants[f"eq{int(PRIMARY_OFFSET_DB)}"] = rhythm_vector(equalized, PRIMARY_OFFSET_DB)
    return variants


def control_vectors(signal: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    """Features that describe the recording rather than the speaker's speech.

    ``channel``: noise floor and SNR. ``silence``: mean cepstrum of non-speech frames.
    A candidate that cannot beat these on held-out microphones is not evidence of a
    disease-specific signal.
    """
    prepared = prepare_signal(signal, sample_rate)
    frames = _frames(prepared)
    energy = np.asarray(20.0 * np.log10(np.sqrt((frames * frames).mean(axis=1)) + 1e-9))
    floor = float(np.percentile(energy, 10))
    channel = np.array([floor, float(np.percentile(energy, 95)) - floor])
    quiet = ~speech_mask(energy, PRIMARY_OFFSET_DB)
    selected = frames[quiet] if quiet.sum() >= 5 else frames[np.argsort(energy)[:5]]
    power = np.abs(np.fft.rfft(selected * _WINDOW, n=512, axis=1)) ** 2
    cepstrum = np.log(np.maximum(power @ _FILTERBANK, 1e-12)) @ _DCT
    silence = cepstrum[:, 1:SILENCE_COEFFICIENTS + 1].mean(axis=0)
    return {"channel": channel, "silence": silence}
