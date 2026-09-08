from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "robot").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    cwd = Path.cwd()
    if (cwd / "robot").is_dir():
        return cwd
    raise FileNotFoundError("Could not locate the quad-loco repo root (expected robot/ and pyproject.toml).")


def robot_dir() -> Path:
    return repo_root() / "robot"


def original_urdf() -> Path:
    return (
        repo_root()
        / "custom_quadruped_isaac"
        / "src"
        / "robot_simulation"
        / "robot_description"
        / "robot"
        / "robot.urdf"
    )


def cleaned_urdf() -> Path:
    return robot_dir() / "urdf" / "quadruped.urdf"


def mesh_dir() -> Path:
    return robot_dir() / "meshes"


def mjcf_dir() -> Path:
    return robot_dir() / "mjcf"


def robot_xml() -> Path:
    return mjcf_dir() / "quadruped.xml"


def scene_xml() -> Path:
    return mjcf_dir() / "scene.xml"
