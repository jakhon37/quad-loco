#!/usr/bin/env python3
"""Train PPO on the custom quadruped (CPU by default, CUDA when available)."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ["PYTHONPATH"] = f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(SRC))
warnings.filterwarnings("ignore", message="Gym has been unmaintained")


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


# def make_env(src: str, easy: bool, seed: int, idx: int, render_mode=None):
#     """Factory whose body runs inside each vec-env worker (forkserver-safe)."""

#     def _init():
#         import sys as _sys

#         if src not in _sys.path:
#             _sys.path.insert(0, src)
#         from quad_loco.env import QuadrupedVelocityEnv

#         env = QuadrupedVelocityEnv(easy=easy, render_mode=render_mode)
#         env.reset(seed=seed + idx)
#         return env

#     return _init
# def make_env(env_id: str, easy: bool, seed: int, idx: int, render_mode=None):
#     def _init():
#         import sys
#         from pathlib import Path

#         src = Path(__file__).resolve().parents[1] / "src"
#         if str(src) not in sys.path:
#             sys.path.insert(0, str(src))

#         import gymnasium as gym
#         import quad_loco  # noqa: F401

#         env = gym.make(env_id, easy=easy, render_mode=render_mode)
#         env.reset(seed=seed + idx)
#         return env

#     return _init


# def make_env(env_id: str, easy: bool, seed: int, idx: int, render_mode=None):
#     def _init():
#         import sys
#         from pathlib import Path

#         root = Path(__file__).resolve().parents[1]
#         src = str(root / "src")
#         if src not in sys.path:
#             sys.path.insert(0, src)

#         import gymnasium as gym
#         import quad_loco  # noqa: F401 - registers env ids as a side effect

#         env = gym.make(env_id, easy=easy, render_mode=render_mode)
#         env.reset(seed=seed + idx)
#         return env

#     return _init
    
def make_env(easy: bool, seed: int, idx: int, render_mode=None):
    def _init():
        import sys
        from pathlib import Path
        # Add the src folder to the path inside the worker process
        src = Path(__file__).resolve().parents[1] / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from quad_loco.env import QuadrupedVelocityEnv
        env = QuadrupedVelocityEnv(easy=easy, render_mode=render_mode)
        env.reset(seed=seed + idx)
        return env
    return _init

    
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ppo_cpu.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--easy", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_envs = args.n_envs or int(cfg.get("n_envs", 1))
    timesteps = args.timesteps or int(cfg.get("total_timesteps", 200_000))
    device = args.device or cfg.get("device", "auto")
    easy = args.easy or bool(cfg.get("easy", False))
    seed = int(cfg.get("seed", 1))
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.config.stem
    src = str(SRC)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

    use_subproc = cfg.get("vec_env") == "subproc" and n_envs > 1
    if use_subproc:
        env = SubprocVecEnv(
            [make_env(src, easy, seed, i) for i in range(n_envs)],
            start_method="forkserver",
        )
    else:
        env = DummyVecEnv([make_env(src, easy, seed, i) for i in range(n_envs)])
    env = VecMonitor(env)
    if cfg.get("normalize", True):
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(src, easy, seed + 10_000, 0)])
    eval_env = VecMonitor(eval_env)
    if cfg.get("normalize", True):
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.obs_rms = env.obs_rms

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        n_steps=int(cfg.get("n_steps", 2048)),
        batch_size=int(cfg.get("batch_size", 256)),
        n_epochs=int(cfg.get("n_epochs", 10)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        ent_coef=float(cfg.get("ent_coef", 0.003)),
        vf_coef=float(cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        policy_kwargs={"net_arch": list(cfg.get("net_arch", [256, 256]))},
        tensorboard_log=str(log_dir / "tb"),
        verbose=1,
        seed=seed,
        device=device,
    )

    ckpt_dir = log_dir / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        CheckpointCallback(save_freq=max(50_000 // n_envs, 1), save_path=str(ckpt_dir), name_prefix="ckpt"),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(ckpt_dir / "eval"),
            eval_freq=max(25_000 // n_envs, 1),
            n_eval_episodes=3,
            deterministic=True,
        ),
    ]
    model.learn(total_timesteps=timesteps, callback=callbacks, tb_log_name=run_name)
    # SB3 appends .zip; pass the stem so we do not get final_model.zip.zip
    model_stem = ckpt_dir / "final_model"
    model.save(model_stem)
    saved = Path(str(model_stem) + ".zip")
    if not saved.is_file() and model_stem.is_file():
        saved = model_stem
    if cfg.get("normalize", True):
        env.save(str(ckpt_dir / "vecnormalize.pkl"))
    print(f"saved {saved}")
    env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
