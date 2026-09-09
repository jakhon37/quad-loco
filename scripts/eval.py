#!/usr/bin/env python3
"""Roll out a trained policy, print metrics, optionally write a video."""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message="Gym has been unmaintained")
warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")

from quad_loco.gl_setup import configure_mujoco_gl, save_rollout_visuals  # noqa: E402

T0 = time.time()
GL = configure_mujoco_gl()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


log(f"python={sys.version.split()[0]}  MUJOCO_GL={GL}")

import numpy as np  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnorm", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--easy", action="store_true")
    parser.add_argument("--command", nargs=3, type=float, default=None, metavar=("VX", "VY", "YAW"))
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--max-seconds", type=float, default=8.0, help="Cap recorded video length")
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    log("importing torch / sb3 / mujoco env …")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from quad_loco.checkpoints import resolve_sb3_zip
    from quad_loco.constants import CONTROL_DT
    from quad_loco.env import QuadrupedVelocityEnv

    model_path = resolve_sb3_zip(args.model)
    log(f"model {model_path}")
    command = tuple(args.command) if args.command else None
    want_video = args.video is not None
    render_mode = "rgb_array" if want_video else None
    max_steps = int(args.max_seconds / CONTROL_DT) if args.video else None

    log("creating env")
    def _make():
        return QuadrupedVelocityEnv(easy=args.easy, command=command, render_mode=render_mode)

    venv = DummyVecEnv([_make])
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        candidate = model_path.parent / "vecnormalize.pkl"
        if candidate.is_file():
            vecnorm_path = candidate
    if vecnorm_path and Path(vecnorm_path).is_file():
        log(f"loading vecnormalize {vecnorm_path}")
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False

    load_arg = str(model_path)
    if load_arg.endswith(".zip"):
        load_arg = load_arg[:-4]
    log("loading PPO (cpu)")
    model = PPO.load(load_arg, env=venv, device="cpu")
    raw = venv.venv.envs[0] if hasattr(venv, "venv") else venv.envs[0]
    if hasattr(raw, "unwrapped"):
        raw = raw.unwrapped
    log("policy loaded")

    render_every = 1
    if want_video:
        log("first render (OSMesa/EGL init can take 10–60s) …")
        t_r = time.time()
        try:
            probe = raw.render()
        except Exception as exc:
            log(f"render failed ({type(exc).__name__}: {exc}) — continuing without video")
            want_video = False
            probe = None
        else:
            dt = time.time() - t_r
            log(f"first frame ok in {dt:.1f}s  shape={getattr(probe, 'shape', None)}")
            if dt > 0.4 and max_steps:
                render_every = max(2, int(dt / 0.15))
                log(f"slow renderer: capturing every {render_every} steps")

    frames = []
    returns = []
    vx_err = []
    for ep in range(args.episodes):
        log(f"episode {ep + 1}/{args.episodes}  cap={max_steps or 'full'} steps")
        obs = venv.reset()
        done = False
        ep_ret = 0.0
        infos = [{}]
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = venv.step(action)
            ep_ret += float(reward[0])
            lin = infos[0].get("lin_vel")
            cmd = infos[0].get("command")
            if lin is not None and cmd is not None:
                vx_err.append(abs(float(lin[0]) - float(cmd[0])))
            if want_video and (max_steps is None or steps < max_steps) and steps % render_every == 0:
                try:
                    frame = raw.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as exc:
                    log(f"render failed at step {steps}: {exc}")
                    want_video = False
            steps += 1
            if steps % 25 == 0 or (max_steps and steps == max_steps):
                h = infos[0].get("base_height")
                tot = max_steps or "?"
                log(f"  step {steps}/{tot}  h={h}  frames={len(frames)}")
            done = bool(dones[0]) or (max_steps is not None and steps >= max_steps)
        returns.append(ep_ret)
        log(f"episode {ep} done  return={ep_ret:.2f}  height={infos[0].get('base_height')}  steps={steps}")

    log(
        f"mean_return={np.mean(returns):.2f}  "
        f"mean_|vx-cmd|={np.mean(vx_err) if vx_err else float('nan'):.3f}"
    )
    if want_video and frames:
        log(f"encoding {len(frames)} frames …")
        written = save_rollout_visuals(frames, args.video)
        for kind, path in written.items():
            log(f"wrote {kind}: {path} ({path.stat().st_size} bytes)")
    elif args.video and not frames:
        log("no frames captured; apt-get install libosmesa6")
    venv.close()
    log("eval done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
