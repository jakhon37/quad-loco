#!/usr/bin/env python3
"""Hold the Isaac Lab home pose with PD and report whether the robot stays up."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quad_loco.constants import CONTROL_DT, FRAME_SKIP, HOME_HEIGHT  # noqa: E402
from quad_loco.paths import scene_xml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--xml", type=Path, default=None)
    args = parser.parse_args()

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

    def run_loop(sync=None):
        for _ in range(n_steps):
            data.ctrl[:] = model.key_ctrl[0]
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            heights.append(float(data.qpos[2]))
            rot = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")].reshape(3, 3)
            gravity = rot.T @ np.array([0.0, 0.0, -1.0])
            tilts.append(float(np.linalg.norm(gravity[:2])))
            if sync is not None:
                sync()

    if args.viewer:
        from mujoco import viewer

        with viewer.launch_passive(model, data) as vis:
            run_loop(sync=vis.sync)
    else:
        run_loop()

    z0 = heights[0]
    z1 = heights[-1]
    mean_z = float(np.mean(heights))
    mean_tilt = float(np.mean(tilts))
    print(f"home_height_target={HOME_HEIGHT:.3f} dt={CONTROL_DT:.3f}s")
    print(f"z_start={z0:.3f} z_end={z1:.3f} z_mean={mean_z:.3f} tilt_xy_mean={mean_tilt:.3f}")
    ok = z1 > 0.35 and mean_tilt < 0.35 and (z0 - z1) < 0.25
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
