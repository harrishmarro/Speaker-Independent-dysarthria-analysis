import csv
import json
import sys

import numpy as np
import pytest
import soundfile as sf

import transfer
from dysarthria.audio_io import load_mono_16k, resample


def tone(frequency: float, seconds: float, rate: int) -> np.ndarray:
    return np.sin(2 * np.pi * frequency * np.arange(int(seconds * rate)) / rate)


def peak_hz(signal: np.ndarray, rate: int = 16000) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    return float(np.fft.rfftfreq(signal.size, 1 / rate)[np.argmax(spectrum)])


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_resample_preserves_duration_and_pitch(rate):
    out = resample(tone(440, 1.0, rate), rate)
    assert out.size == 16000
    assert peak_hz(out) == pytest.approx(440, abs=2)


def test_downsampling_removes_content_above_new_nyquist():
    out = resample(tone(440, 1.0, 44100) + tone(12000, 1.0, 44100), 44100)
    spectrum = np.abs(np.fft.rfft(out))
    assert spectrum.max() > 1000 * spectrum[np.fft.rfftfreq(out.size, 1 / 16000) > 7900].max()


def test_load_mono_16k_mixes_channels(tmp_path):
    stereo = np.stack((tone(300, 0.5, 44100), tone(300, 0.5, 44100)), axis=1) * 0.5
    sf.write(tmp_path / "s.wav", stereo, 44100)
    assert load_mono_16k(tmp_path / "s.wav").size == 8000


def speech(bursts: int, noise: float, seed: int, rate: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    voiced = 0.5 * tone(180 + 7 * seed % 60, 0.18, rate)
    gap = np.zeros(int(0.25 * rate))
    signal = np.concatenate([gap] + [np.concatenate((voiced, gap)) for _ in range(bursts)])
    return signal + noise * rng.standard_normal(signal.size)


def write_corpus(directory, prefix, rate, confounded, split=True, speakers=10):
    rows = []
    for index in range(speakers):
        label = "dysarthria" if index % 2 else "control"
        speaker = f"{prefix}{index:02d}"
        for take in range(3):
            # Disease signal: dysarthric speakers pause more (more, shorter bursts).
            bursts = 7 if label == "dysarthria" else 3
            # Source corpus: noise level also tracks the label (a recording confound).
            # Target corpus: noise level is random, so the confound does not transfer.
            noise = (0.02 if label == "dysarthria" else 0.001) if confounded else (
                [0.001, 0.02][(index // 2 + take) % 2])  # both classes get both levels
            path = directory / f"{speaker}_{take}.wav"
            sf.write(path, speech(bursts, noise, index * 10 + take, rate), rate)
            row = {"audio_path": str(path), "speaker_id": speaker, "label": label}
            if split:
                row["split"] = "test" if index >= speakers - 2 else "train"
            rows.append(row)
    manifest = directory / f"{prefix}.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_frozen_transfer_end_to_end(tmp_path, monkeypatch, capsys):
    source = write_corpus(tmp_path, "S", 16000, confounded=True)
    target = write_corpus(tmp_path, "T", 44100, confounded=False, split=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["transfer.py", "--source-manifest", str(source),
                                      "--target-manifest", str(target),
                                      "--candidate", "rhythm", "--output-dir", str(out)])
    transfer.main()
    summary = json.loads((out / "transfer_summary.json").read_text())
    # Source test speakers (S08, S09) stay locked: 8 of 10 used; all 10 target speakers.
    assert summary["source_speakers"] == 8 and summary["target_speakers"] == 10
    assert summary["results"]["rhythm_transfer"]["auc"] == 1.0
    # The source noise confound does not survive transfer to a corpus without it.
    assert summary["results"]["channel_transfer"]["auc"] < 0.9
    assert set(summary["auc_gain_over_controls"]) == {
        "channel_transfer", "channel_within_target", "silence_transfer",
        "silence_within_target"}
    assert summary["verdict"] in {"pass", "fail"} and summary["reason"]
    assert json.loads(capsys.readouterr().out.strip())["candidate_auc"] == 1.0


def test_inverted_control_is_oriented_before_comparison():
    labels = {"a": 1, "b": 1, "c": 0, "d": 0}
    candidate = {"a": 2.0, "b": 1.0, "c": -1.0, "d": -2.0}
    inverted_control = {"a": -2.0, "b": -1.0, "c": 1.0, "d": 2.0}
    gain, bound = transfer.paired_gain_bound(candidate, inverted_control, labels)
    assert gain == pytest.approx(0.0) and bound <= 0.0


def test_source_manifest_must_be_prepared(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("audio_path,speaker_id,label\nx.wav,S1,control\n")
    with pytest.raises(ValueError, match="split column"):
        transfer.read_rows(manifest, source=True)


def test_transfer_passes_when_signal_is_real_and_target_is_large_enough(tmp_path, monkeypatch):
    # Rhythm differs by class; recording noise is assigned independently of the label.
    source = write_corpus(tmp_path, "S", 16000, confounded=False, speakers=12)
    target = write_corpus(tmp_path, "T", 22050, confounded=False, split=False, speakers=40)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["transfer.py", "--source-manifest", str(source),
                                      "--target-manifest", str(target),
                                      "--candidate", "rhythm", "--output-dir", str(out)])
    transfer.main()
    summary = json.loads((out / "transfer_summary.json").read_text())
    assert summary["verdict"] == "pass", summary["reason"]
    assert all(v["lower_95"] > 0 for v in summary["auc_gain_over_controls"].values())
