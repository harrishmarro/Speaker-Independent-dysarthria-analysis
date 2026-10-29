from unittest.mock import patch

import pytest

import verify_runner


def test_runner_rejects_missing_manifest(tmp_path):
    with patch("verify_runner.available_accelerator", return_value="mps:Apple Metal"):
        with patch("verify_runner.argparse.ArgumentParser.parse_args") as arguments:
            arguments.return_value.manifest = str(tmp_path / "missing.csv")
            with pytest.raises(SystemExit, match="does not exist"):
                verify_runner.main()


def test_runner_rejects_sample_manifest():
    sample = "data/sample_manifest.csv"
    with patch("verify_runner.available_accelerator", return_value="mps:Apple Metal"):
        with patch("verify_runner.argparse.ArgumentParser.parse_args") as arguments:
            arguments.return_value.manifest = sample
            with pytest.raises(SystemExit, match="refuses the synthetic"):
                verify_runner.main()


def test_acoustic_candidate_can_run_without_accelerator(tmp_path, capsys):
    manifest = tmp_path / "splits.clean.csv"
    manifest.touch()
    with patch("verify_runner.available_accelerator", return_value=None):
        with patch("verify_runner.argparse.ArgumentParser.parse_args") as arguments:
            arguments.return_value.manifest = str(manifest)
            arguments.return_value.allow_cpu = True
            verify_runner.main()
    assert "cpu (acoustic candidate)" in capsys.readouterr().out
