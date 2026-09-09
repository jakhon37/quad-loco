#!/usr/bin/env python3
"""Hold the default pose with PD and report whether the robot stays up."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quad_loco.constants import CONTROL_DT, FRAME_SKIP, HOME_HEIGHT  # noqa: E402
from quad_loco.paths import scene_xml  # noqa: E402


def _reexec_mjpython() -> None:
    """Cocoa requires the MuJoCo GUI on the main thread; mjpython does that."""
    if sys.platform != "darwin":
        return
    if "mjpython" in Path(sys.executable).name:
        return
    mjpython = shutil.which("mjpython")
    if mjpython is None:
        sibling = Path(sys.executable).resolve().parent / "mjpython"
        if sibling.is_file():
            mjpython = str(sibling)
    if mjpython is None:
        raise SystemExit(
            "On macOS the interactive viewer must be started with mjpython "
            "(not python):\n"
            "  mjpython scripts/stand.py --viewer"
        )
    os.execv(mjpython, [mjpython, *sys.argv])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--xml", type=Path, default=None)
    args = parser.parse_args()

    if args.viewer:
        _reexec_mjpython()

    import mujoco

    xml = args.xml or scene_xml()
    if not xml.is_file():
        from quad_loco.convert_urdf import convert

        xml = convert()

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    n_steps = int(args.seconds / (model.opt.timestep * FRAME_SKIP))
    heights = []
    tilts = []

    def sample_stats() -> None:
        heights.append(float(data.qpos[2]))
        rot = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")].reshape(3, 3)
        gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        tilts.append(float(np.linalg.norm(gravity[:2])))

    def physics_step(*, record: bool) -> None:
        data.ctrl[:] = model.key_ctrl[0]
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)
        if record:
            sample_stats()

    if args.viewer:
        from mujoco import viewer

        dt = model.opt.timestep * FRAME_SKIP
        print("Opening MuJoCo window. Close it (or Ctrl+C) to exit.", flush=True)
        print(
            "On macOS, mjpython may print 'Task policy set failed'; that is a "
            "Cocoa thread-QoS warning and can be ignored if the window is open.",
            flush=True,
        )
        try:
            with viewer.launch_passive(model, data) as vis:
                while vis.is_running():
                    t0 = time.perf_counter()
                    physics_step(record=len(heights) < n_steps)
                    vis.sync()
                    leftover = dt - (time.perf_counter() - t0)
                    if leftover > 0:
                        time.sleep(leftover)
        except KeyboardInterrupt:
            print("\nviewer interrupted", flush=True)
    else:
        for _ in range(n_steps):
            physics_step(record=True)

    if not heights:
        print("no samples")
        return 1
    z0 = heights[0]
    z1 = heights[min(len(heights) - 1, n_steps - 1)]
    mean_z = float(np.mean(heights[:n_steps] if len(heights) >= n_steps else heights))
    mean_tilt = float(np.mean(tilts[:n_steps] if len(tilts) >= n_steps else tilts))
    print(f"home_height_target={HOME_HEIGHT:.3f} dt={CONTROL_DT:.3f}s")
    print(f"z_start={z0:.3f} z_end={z1:.3f} z_mean={mean_z:.3f} tilt_xy_mean={mean_tilt:.3f}")
    ok = z1 > 0.35 and mean_tilt < 0.35 and (z0 - z1) < 0.25
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
