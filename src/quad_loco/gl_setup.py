"""Pick a MuJoCo GL backend before `import mujoco`. Headless Colab has no DISPLAY."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_mujoco_gl() -> str:
    if "MUJOCO_GL" in os.environ:
        return os.environ["MUJOCO_GL"]
    if sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "cgl"
        return "cgl"
    if os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "glfw"
        return "glfw"
    # Colab / SSH Linux: OSMesa is the reliable software path.
    # EGL is faster on a GPU but often missing libEGL in the runtime.
    os.environ["MUJOCO_GL"] = "osmesa"
    return "osmesa"


def save_rollout_visuals(frames: list, mp4_path: Path, fps: int = 50) -> dict[str, Path]:
    import imageio.v2 as imageio

    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    mid = frames[len(frames) // 2]
    preview = mp4_path.with_name(mp4_path.stem + "_preview.png")
    imageio.imwrite(preview, mid)
    written["png"] = preview

    try:
        imageio.mimsave(mp4_path, frames, fps=fps)
        written["mp4"] = mp4_path
    except Exception as exc:
        print(f"mp4 skipped ({exc}); writing a small gif instead")
        gif = mp4_path.with_suffix(".gif")
        step = max(1, len(frames) // 24)
        small = [f[::2, ::2] for f in frames[::step]]
        imageio.mimsave(gif, small, fps=12, loop=0)
        written["gif"] = gif
    return written
