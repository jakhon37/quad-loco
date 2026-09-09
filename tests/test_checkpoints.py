from __future__ import annotations

from pathlib import Path

import pytest

from quad_loco.checkpoints import resolve_sb3_zip


def test_resolve_strips_double_zip(tmp_path: Path):
    model = tmp_path / "final_model.zip"
    model.write_bytes(b"zip")
    assert resolve_sb3_zip(tmp_path / "final_model.zip.zip") == model.resolve()
    assert resolve_sb3_zip(tmp_path / "final_model") == model.resolve()


def test_resolve_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="--run-name must match"):
        resolve_sb3_zip(tmp_path / "final_model.zip")
