#!/usr/bin/env python3
"""Export a Stable-Baselines3 PPO actor to ONNX."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message="Gym has been unmaintained")
warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")

from quad_loco.runtime import RuntimeClock  # noqa: E402

CLOCK = RuntimeClock(heartbeat_s=15)


def log(msg: str) -> None:
    CLOCK.log(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    log("importing sb3")
    from stable_baselines3 import PPO

    from quad_loco.checkpoints import resolve_sb3_zip
    from quad_loco.export import export_sb3_onnx, verify_onnx

    model_path = resolve_sb3_zip(args.model)
    load_arg = str(model_path)
    if load_arg.endswith(".zip"):
        load_arg = load_arg[:-4]
    log(f"loading {load_arg} on cpu")
    model = PPO.load(load_arg, device="cpu")
    out = args.out or model_path.with_name("policy.onnx")
    log(f"tracing ONNX -> {out}")
    export_sb3_onnx(model, out)
    log("verifying ONNX vs PyTorch")
    dummy = np.zeros(48, dtype=np.float32)
    err = verify_onnx(out, model, dummy)
    log(f"wrote {out}  max_abs_err={err:.6g}")
    CLOCK.stop()
    return 0 if err < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
