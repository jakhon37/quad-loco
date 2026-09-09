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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    from quad_loco.checkpoints import resolve_sb3_zip
    from quad_loco.export import export_sb3_onnx, verify_onnx

    model_path = resolve_sb3_zip(args.model)
    load_arg = str(model_path)
    if load_arg.endswith(".zip"):
        load_arg = load_arg[:-4]
    model = PPO.load(load_arg)
    out = args.out or model_path.with_name("policy.onnx")
    export_sb3_onnx(model, out)
    dummy = np.zeros(48, dtype=np.float32)
    err = verify_onnx(out, model, dummy)
    print(f"wrote {out}  max_abs_err={err:.6g}")
    return 0 if err < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
