import csv
import json
import sys

import numpy as np
import pytest
import soundfile as sf

import controls
import evaluate
import train
from dysarthria import rhythm_features as rf
from dysarthria.acoustic_cv import mic_swap
from locked import candidate_io

SR = 16000


def bursts(count: int, burst_s: float = 0.2, gap_s: float = 0.3, noise: float = 0.0,
           seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(int(burst_s * SR)) / SR)
    gap = np.zeros(int(gap_s * SR))
    signal = np.concatenate([gap] + [np.concatenate((tone, gap)) for _ in range(count)])
    return signal + noise * rng.standard_normal(signal.size)


def test_rhythm_counts_pauses_inside_trimmed_speech_region():
    occupancy, pauses, run, _ = rf.rhythm_vector(
        rf.frame_energy_db(rf.prepare_signal(bursts(3), SR)), 15.0)
    assert pauses == pytest.approx(2 / 1.2, rel=0.1)  # 2 internal pauses over ~1.2 s
    assert run == pytest.approx(0.2, abs=0.03)
    assert occupancy == pytest.approx(0.6 / 1.2, abs=0.05)


def test_rhythm_ignores_gain_and_a_low_noise_floor():
    reference = rf.rhythm_variants(bursts(4), SR, "a")["vad15"]
    quieter_noisier = rf.rhythm_variants(0.1 * bursts(4, noise=0.0005), SR, "a")["vad15"]
    assert np.allclose(reference[:3], quieter_noisier[:3], rtol=0.1, atol=0.02)


def test_equalization_is_deterministic_and_sets_a_common_floor():
    signal = rf.prepare_signal(bursts(2), SR)
    assert np.array_equal(rf.equalize(signal, "x.wav"), rf.equalize(signal, "x.wav"))
    floor = np.percentile(rf.frame_energy_db(rf.equalize(signal, "x.wav")), 10)
    assert floor == pytest.approx(rf.EQUALIZED_FLOOR_DB, abs=1.5)


def test_controls_describe_the_recording_not_the_rhythm():
    quiet = rf.control_vectors(bursts(3, noise=0.0001), SR)["channel"]
    noisy = rf.control_vectors(bursts(3, noise=0.01), SR)["channel"]
    assert noisy[0] > quiet[0] + 20  # noise floor rises
    assert noisy[1] < quiet[1]  # SNR falls


def test_mic_swap_fits_on_source_mic_only_and_skips_single_mic_speakers():
    labels = {"C1": "control", "C2": "control", "C3": "control",
              "D1": "dysarthria", "D2": "dysarthria", "D3": "dysarthria", "D4": "dysarthria"}
    features = {}
    for speaker, label in labels.items():
        signal = 1.0 if label == "dysarthria" else -1.0
        features[(speaker, "headMic")] = np.array([signal, 5.0 * signal])
        # a channel cue that reverses on the other microphone
        features[(speaker, "arrayMic")] = np.array([signal, -5.0 * signal])
    del features[("D4", "arrayMic")]
    settings = {"classifier": "ridge", "ridge_alpha": 1.0, "spectral_weight": 1.0,
                "feature_set": "channel"}
    results = mic_swap(features, labels, settings, "headMic", "arrayMic")
    assert {row["speaker_id"] for row in results} == set(labels) - {"D4"}
    with pytest.raises(ValueError, match="different target"):
        mic_swap(features, labels, settings, "headMic", "headMic")


def write_manifest(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_both_mics_split_expects_only_dual_mic_speakers(tmp_path):
    rows = [{"audio_path": f"{s}{m}.wav", "speaker_id": s, "label": label, "mic": m,
             "split": "train"}
            for s, label, mics in (("C1", "control", ("headMic", "arrayMic")),
                                   ("D1", "dysarthria", ("headMic", "arrayMic")),
                                   ("D2", "dysarthria", ("headMic",)))
            for m in mics]
    write_manifest(tmp_path / "m.csv", rows)
    assert evaluate.expected_labels(tmp_path / "m.csv", "development", both_mics=True) == {
        "C1": 0, "D1": 1}


def test_train_and_controls_write_every_swap_file(tmp_path, monkeypatch):
    root = tmp_path / "project"
    configs = root / "configs" / "research"
    configs.mkdir(parents=True)
    (configs / "active.json").write_text(json.dumps({
        "candidate": "rhythm_ridge_1", "feature_set": "rhythm", "classifier": "ridge",
        "ridge_alpha": 1.0, "spectral_weight": 1.0, "aggregation": "median"}))
    (configs / "baseline.json").write_text(json.dumps({
        "candidate": "spectral_equal", "spectral_weight": 1.0, "aggregation": "mean"}))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(candidate_io, "ARTIFACTS", artifacts)
    monkeypatch.setattr(train, "__file__", str(root / "train.py"))
    rows = []
    speakers = [(f"C{i}", "control", 3) for i in range(1, 4)] + [
        (f"D{i}", "dysarthria", 5) for i in range(1, 4)]
    for index, (speaker, label, count) in enumerate(speakers):
        for mic in ("headMic", "arrayMic"):
            for take in range(2):
                path = tmp_path / f"{speaker}_{mic}_{take}.wav"
                noise = 0.001 if mic == "headMic" else 0.01
                sf.write(path, bursts(count, noise=noise, seed=index * 10 + take), SR)
                rows.append({"audio_path": str(path), "speaker_id": speaker, "label": label,
                             "mic": mic, "split": "validation" if index in (2, 5) else "train"})
    manifest = artifacts / "candidate.csv"
    write_manifest(manifest, rows)
    output = artifacts / "predictions.jsonl"
    monkeypatch.setattr(sys, "argv", ["train.py", "--manifest", str(manifest),
                                      "--output", str(output)])
    train.main()
    for name in ("vad10", "vad15", "vad20", "eq15", "baseline"):
        for direction in ("h2a", "a2h"):
            lines = (artifacts / f"swap_{name}_{direction}.jsonl").read_text().splitlines()
            assert len(lines) == 6
    monkeypatch.setattr(sys, "argv", ["controls.py", "--manifest", str(manifest),
                                      "--output-dir", str(artifacts)])
    controls.main()
    for name in ("channel", "silence"):
        for suffix in ("h2a", "a2h", "cv"):
            assert (artifacts / f"control_{name}_{suffix}.jsonl").exists()


def test_controls_refuse_test_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate_io, "ARTIFACTS", tmp_path)
    write_manifest(tmp_path / "m.csv", [{"audio_path": "x.wav", "speaker_id": "T1",
                                         "label": "control", "mic": "headMic",
                                         "split": "test"}])
    monkeypatch.setattr(sys, "argv", ["controls.py", "--manifest", str(tmp_path / "m.csv"),
                                      "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="test rows"):
        controls.main()


def test_torgo_builder_records_mic_session_and_take(tmp_path, monkeypatch):
    sys.path.insert(0, "scripts")
    import build_torgo_manifest

    for folder, name in (("F_Dys/wav_headMic_F03S02", "wav_headMic_F03S02_0007.wav"),
                         ("F_Dys/wav_arrayMic_F03S02", "wav_arrayMic_F03S02_0007.wav"),
                         ("M_Con/wav_headMic_MC01", "wav_headMic_MC01_0001.wav")):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder / name).touch()
    output = tmp_path / "out.csv"
    monkeypatch.setattr(sys, "argv", ["build", str(tmp_path), str(output)])
    build_torgo_manifest.main()
    with output.open() as handle:
        rows = {(r["mic"], r["speaker_id"]): r for r in csv.DictReader(handle)}
    assert rows[("headMic", "F03")]["take_id"] == rows[("arrayMic", "F03")]["take_id"]
    assert rows[("headMic", "F03")]["session"] == "S02"
    assert rows[("headMic", "MC01")]["session"] == "S00"
