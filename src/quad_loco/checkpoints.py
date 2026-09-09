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

    found = []
    for root in (parent, Path("logs"), Path(__file__).resolve().parents[2] / "logs"):
        if root.is_dir():
            found.extend(sorted(root.glob("**/final_model.zip")))
    found = list(dict.fromkeys(found))
    hint = ""
    if found:
        newest = max(found, key=lambda p: p.stat().st_mtime)
        hint = f"\nAvailable runs:\n  " + "\n  ".join(str(p) for p in found)
        hint += f"\nNewest: {newest}"
    raise FileNotFoundError(
        f"No trained model at {path}. --run-name must match the train command "
        f"(e.g. smoke → logs/smoke/final_model.zip).\n"
        f"Looked at: {', '.join(str(p) for p in tried)}"
        f"{hint}"
    )
