"""Load any corpus's audio as mono 16 kHz float64, the rate every feature module expects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_RATE = 16000


def resample(signal: np.ndarray, rate: int, target: int = TARGET_RATE) -> np.ndarray:
    """Band-limited FFT resampling of a whole recording.

    Downsampling keeps only bins below the new Nyquist frequency (an ideal low-pass), so
    there is no aliasing. Deterministic and numpy-only.
    """
    if rate <= 0:
        raise ValueError("sample rate must be positive")
    if rate == target or signal.size == 0:
        return np.asarray(signal, dtype=np.float64)
    count = max(1, round(signal.size * target / rate))
    spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
    bins = count // 2 + 1
    if spectrum.size >= bins:
        spectrum = spectrum[:bins]
    else:
        spectrum = np.concatenate((spectrum, np.zeros(bins - spectrum.size, dtype=complex)))
    return np.asarray(np.fft.irfft(spectrum, n=count) * (count / signal.size))


def load_mono_16k(path: str | Path) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float64", always_2d=False)
    signal = np.asarray(audio)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return resample(signal, int(rate))
