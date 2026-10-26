from pathlib import Path

import pytest

from locked.candidate_io import read_prepared_manifest


def test_candidate_cannot_read_outside_artifacts(tmp_path):
    outside = tmp_path / "private.csv"
    outside.write_text("secret\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must remain within"):
        read_prepared_manifest(outside)


def test_candidate_cannot_escape_with_parent_segments():
    with pytest.raises(ValueError, match="must remain within"):
        read_prepared_manifest(Path("artifacts") / ".." / "README.md")
