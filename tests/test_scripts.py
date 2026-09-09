from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "check.py",
    ROOT / "scripts" / "eval.py",
    ROOT / "scripts" / "export_onnx.py",
    ROOT / "scripts" / "train.py",
    ROOT / "scripts" / "stand.py",
    ROOT / "scripts" / "smoke.py",
]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_scripts_compile(path: Path):
    py_compile.compile(str(path), doraise=True)


def test_package_modules_compile():
    for path in (ROOT / "src" / "quad_loco").glob("*.py"):
        py_compile.compile(str(path), doraise=True)
