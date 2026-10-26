from pathlib import Path

from security import inspect_candidate


def test_current_candidate_passes_policy():
    assert inspect_candidate(Path("train.py")) == []


def test_network_import_is_rejected(tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("import requests\n", encoding="utf-8")
    assert "forbidden import requests" in inspect_candidate(candidate)[0]


def test_dynamic_execution_is_rejected(tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("eval('1 + 1')\n", encoding="utf-8")
    assert "forbidden call eval" in inspect_candidate(candidate)[0]
