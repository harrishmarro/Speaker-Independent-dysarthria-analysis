"""Check that TORGO condition auditing preserves speaker separation and privacy."""

import pytest

from scripts.audit_torgo_conditions import audit


def row(group: str, directory: str, speaker: str, label: str, split: str) -> dict[str, str]:
    return {
        "audio_path": f"/private/archive/{group}/{directory}/{directory}_0001.wav",
        "speaker_id": speaker,
        "label": label,
        "split": split,
    }


def test_counts_sessions_and_microphones_without_identifiers() -> None:
    rows = [
        row("F_Con", "wav_arrayMic_FC01S01", "FC01", "control", "train"),
        row("F_Con", "wav_headMic_FC01S01", "FC01", "control", "train"),
        row("F_Con", "wav_arrayMic_FC01S02", "FC01", "control", "train"),
        row("M_Dys", "wav_headMic_M01", "M01", "dysarthria", "validation"),
    ]
    report = audit(rows)
    group = report["by_split_and_label"]["train"]["control"]
    assert group == {
        "recordings": 3, "speakers": 1,
        "microphones": {"arrayMic": 2, "headMic": 1},
        "speaker_sessions_with_id": 2, "unspecified_session_recordings": 0,
    }
    assert "FC01" not in str(report)
    assert "/private" not in str(report)


def test_rejects_cross_split_speaker() -> None:
    rows = [
        row("F_Con", "wav_arrayMic_FC01S01", "FC01", "control", "train"),
        row("F_Con", "wav_headMic_FC01S02", "FC01", "control", "validation"),
    ]
    with pytest.raises(ValueError, match="multiple splits"):
        audit(rows)


def test_rejects_label_or_speaker_mismatch() -> None:
    invalid = row("F_Dys", "wav_arrayMic_F03S01", "F03", "control", "train")
    with pytest.raises(ValueError, match="conflicts"):
        audit([invalid])
