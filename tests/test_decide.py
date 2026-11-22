import json
import sys

from autoresearch import decide as controller
from autoresearch.decide import gate_from_record, swap_gate

CONTROLS = {"channel": {"h2a": 0.40, "a2h": 0.55}, "silence": {"h2a": 0.35, "a2h": 0.45}}
BASELINE = {"h2a": 0.50, "a2h": 0.60}


def test_swap_gate_requires_every_variant_both_directions_and_control_margin():
    passing = {"vad10": {"h2a": 0.80, "a2h": 0.75}, "vad15": {"h2a": 0.78, "a2h": 0.72}}
    assert swap_gate(passing, CONTROLS, BASELINE)[0]
    one_weak_variant = {**passing, "vad20": {"h2a": 0.80, "a2h": 0.65}}
    ok, reason = swap_gate(one_weak_variant, CONTROLS, BASELINE)
    assert not ok and "vad20 a2h" in reason
    near_control = {"main": {"h2a": 0.80, "a2h": 0.70}}
    assert not swap_gate(near_control, {**CONTROLS, "channel": {"h2a": 0.4, "a2h": 0.6}},
                         BASELINE)[0]
    assert not swap_gate(passing, CONTROLS, {"h2a": 0.9, "a2h": 0.9})[0]


def test_gate_refuses_results_without_mic_swap_fields():
    import pytest
    with pytest.raises(ValueError, match="mic-provenance"):
        gate_from_record({"swap_main_h2a_ba": ""}, "spectral")


def test_new_approach_waits_for_explicit_approval(tmp_path, monkeypatch, capsys):
    state = tmp_path / "state.json"
    state.write_text('{"next_index": 3}')
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"balanced_accuracy": 0.5, "f1": 0.0}')
    approvals = tmp_path / "approvals.json"
    approvals.write_text('{"approved_approaches": ["centroid", "speaker_knn"]}')
    active = tmp_path / "active.json"
    monkeypatch.setattr(controller, "STATE", state)
    monkeypatch.setattr(controller, "ACCEPTED", baseline)
    monkeypatch.setattr(controller, "APPROVALS", approvals)
    monkeypatch.setattr(controller, "ACTIVE", active)
    monkeypatch.setattr(sys, "argv", ["decide.py", "prepare"])

    controller.main()
    assert "review_required=true" in capsys.readouterr().out
    assert not active.exists()
    assert json.loads(state.read_text())["next_index"] == 3

    approvals.write_text('{"approved_approaches": ["centroid", "speaker_knn", "ridge"]}')
    controller.main()
    assert "candidate_available=true" in capsys.readouterr().out
    assert json.loads(active.read_text())["candidate"] == "ridge_0_1"

    state.write_text('{"next_index": 6}')
    controller.main()
    assert "review_reason=approach_approval_required" in capsys.readouterr().out
    approvals.write_text(
        '{"approved_approaches": ["centroid", "speaker_knn", "ridge", "mfcc"]}'
    )
    controller.main()
    assert "candidate_available=true" in capsys.readouterr().out
    assert json.loads(active.read_text())["candidate"] == "mfcc_ridge_0_1"
