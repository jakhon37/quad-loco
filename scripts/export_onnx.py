#!/usr/bin/env python3
"""Export a Stable-Baselines3 PPO actor to ONNX."""

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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    from quad_loco.export import export_sb3_onnx, verify_onnx

    model = PPO.load(args.model)
    out = args.out or args.model.with_suffix(".onnx")
    export_sb3_onnx(model, out)
    dummy = np.zeros(48, dtype=np.float32)
    err = verify_onnx(out, model, dummy)
    print(f"wrote {out}  max_abs_err={err:.6g}")
    return 0 if err < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
