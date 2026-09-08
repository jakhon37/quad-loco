#!/usr/bin/env python3
"""End-to-end CPU smoke: convert URDF, env step, two PPO updates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from quad_loco.convert_urdf import convert
    from quad_loco.env import QuadrupedVelocityEnv
    from quad_loco.paths import scene_xml

    xml = convert()
    print(f"scene={xml}")

    env = QuadrupedVelocityEnv(xml_path=xml, easy=True, command=(0.4, 0.0, 0.0))
    obs, _ = env.reset(seed=0)
    info = {"base_height": 0.0}
    reward = 0.0
    for _ in range(25):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"env_ok height={info['base_height']:.3f} reward={reward:.3f}")
    env.close()

    def make():
        return QuadrupedVelocityEnv(xml_path=scene_xml(), easy=True)

    venv = DummyVecEnv([make])
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=64, n_epochs=1, verbose=0)
    model.learn(total_timesteps=128)
    out = ROOT / "logs" / "smoke" / "smoke_model.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"ppo_ok saved {out}")
    venv.close()
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
