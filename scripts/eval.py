#!/usr/bin/env python3
"""Roll out a trained policy, print metrics, optionally write a video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnorm", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--easy", action="store_true")
    parser.add_argument("--command", nargs=3, type=float, default=None, metavar=("VX", "VY", "YAW"))
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    import gymnasium as gym
    import imageio.v2 as imageio
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    import quad_loco  # noqa: F401

    env_id = "QuadVelocityEasy-v0" if args.easy else "QuadVelocity-v0"
    command = tuple(args.command) if args.command else None
    render_mode = "rgb_array" if args.video else None

    def _make():
        return gym.make(env_id, easy=args.easy, command=command, render_mode=render_mode)

    venv = DummyVecEnv([_make])
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        candidate = args.model.parent / "vecnormalize.pkl"
        if candidate.is_file():
            vecnorm_path = candidate
    if vecnorm_path and Path(vecnorm_path).is_file():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(args.model, env=venv)
    raw = venv.venv.envs[0] if hasattr(venv, "venv") else venv.envs[0]
    if hasattr(raw, "unwrapped"):
        raw = raw.unwrapped

    frames = []
    returns = []
    vx_err = []
    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = venv.step(action)
            ep_ret += float(reward[0])
            lin = infos[0].get("lin_vel")
            cmd = infos[0].get("command")
            if lin is not None and cmd is not None:
                vx_err.append(abs(float(lin[0]) - float(cmd[0])))
            if args.video:
                frames.append(raw.render())
            done = bool(dones[0])
        returns.append(ep_ret)
        print(f"episode {ep}: return={ep_ret:.2f} height={infos[0].get('base_height')}")

    print(f"mean_return={np.mean(returns):.2f} mean_|vx-cmd|={np.mean(vx_err) if vx_err else float('nan'):.3f}")
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=50)
        print(f"wrote {args.video}")
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
