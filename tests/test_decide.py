import json
import sys

from autoresearch import decide as controller
from autoresearch.decide import decide


def test_only_strict_primary_gain_without_f1_regression_is_promoted():
    assert not decide(0.5, 0.2, 0.5, 0.0, 0.75, 0.2, 0.5, 0.0)
    assert not decide(0.75, 0.1, 0.5, 0.2, 0.75, 0.2, 0.5, 0.0)
    assert not decide(0.75, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.0)
    assert decide(0.75, 0.2, 0.5, 0.2, 0.75, 0.2, 0.5, 0.0)


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
