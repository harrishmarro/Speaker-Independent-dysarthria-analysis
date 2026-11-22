import json
import sys

from autoresearch import decide as controller
from autoresearch.decide import gate_from_record, swap_gate

CONTROLS = {"channel": {"h2a": 0.40, "a2h": 0.55}, "silence": {"h2a": 0.35, "a2h": 0.45}}
CONTROL_AUC = {"channel": {"h2a": 0.60, "a2h": 0.62}, "silence": {"h2a": 0.55, "a2h": 0.40}}
BASELINE = {"h2a": 0.50, "a2h": 0.60}
PASSING = {"vad10": {"h2a": 0.80, "a2h": 0.75}, "vad15": {"h2a": 0.78, "a2h": 0.72}}
PASSING_AUC = {"vad10": {"h2a": 0.85, "a2h": 0.80}, "vad15": {"h2a": 0.82, "a2h": 0.79}}


def gate(ba=PASSING, auc=PASSING_AUC, controls=CONTROLS, control_auc=CONTROL_AUC,
         baseline=BASELINE):
    return swap_gate(ba, controls, baseline, candidate_auc=auc, control_auc=control_auc)


def test_swap_gate_requires_every_variant_both_directions_and_control_margin():
    assert gate()[0]
    ok, reason = gate(ba={**PASSING, "vad20": {"h2a": 0.80, "a2h": 0.65}},
                      auc={**PASSING_AUC, "vad20": {"h2a": 0.9, "a2h": 0.9}})
    assert not ok and "vad20 a2h" in reason
    assert not gate(controls={**CONTROLS, "channel": {"h2a": 0.4, "a2h": 0.65}})[0]
    assert not gate(baseline={"h2a": 0.9, "a2h": 0.9})[0]


def test_swap_gate_requires_auc_margin_over_controls():
    # Mirrors EXP-20261122T080254Z-0c4e3d: the silence control ranks speakers at AUC
    # 0.857 even though its threshold does not transfer (BA 0.5).
    ok, reason = gate(control_auc={**CONTROL_AUC, "silence": {"h2a": 0.857, "a2h": 0.857}})
    assert not ok and "control AUC 0.857" in reason
    ok, reason = gate(auc={**PASSING_AUC, "vad15": {"h2a": 0.82, "a2h": 0.70}})
    assert not ok and "AUC 0.700 < 0.75" in reason


def test_inverted_control_ranking_still_counts_against_the_candidate():
    # A control at AUC 0.15 ranks speakers as well as one at 0.85, just reversed.
    ok, reason = gate(control_auc={**CONTROL_AUC, "channel": {"h2a": 0.15, "a2h": 0.6}})
    assert not ok and "control AUC 0.850" in reason


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
