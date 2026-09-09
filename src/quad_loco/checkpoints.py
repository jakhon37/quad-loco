from __future__ import annotations

from pathlib import Path


def resolve_sb3_zip(path: Path | str) -> Path:
    """Return an existing SB3 zip. Raises a clear error if training did not save one."""
    path = Path(path)
    raw = str(path)
    while raw.endswith(".zip.zip"):
        raw = raw[:-4]
    path = Path(raw)

    tried: list[Path] = []
    candidates = [path]
    if path.suffix != ".zip":
        candidates.append(Path(str(path) + ".zip"))
    else:
        candidates.append(path.with_suffix(""))

    parent = path.parent if path.parent.as_posix() != "" else Path(".")
    candidates.extend(
        [
            parent / "final_model.zip",
            parent / "final_model",
            parent / "best" / "best_model.zip",
            parent / "best" / "best_model",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        tried.append(candidate)
        if candidate.is_file():
            return candidate
        zipped = Path(str(candidate) + ".zip")
        if zipped.is_file():
            return zipped

    found = sorted(parent.glob("**/*.zip")) if parent.is_dir() else []
    raise FileNotFoundError(
        f"No trained model at {path}. Training probably failed before saving.\n"
        f"Looked at: {', '.join(str(p) for p in tried)}\n"
        f"Zip files present: {', '.join(str(p) for p in found) if found else '(none)'}"
    )
