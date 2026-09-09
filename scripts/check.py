#!/usr/bin/env python3
"""Timed environment check. No trained policy required.

    python -u scripts/check.py
    python -u scripts/check.py --render
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from quad_loco.gl_setup import configure_mujoco_gl  # noqa: E402
from quad_loco.runtime import RuntimeClock  # noqa: E402

CLOCK = RuntimeClock(heartbeat_s=15)
GL = configure_mujoco_gl()
FAILED = 0


def log(msg: str) -> None:
    CLOCK.log(msg)


def ok(name: str, detail: str = "") -> None:
    extra = f"  {detail}" if detail else ""
    log(f"OK   {name}{extra}")


def fail(name: str, exc: BaseException) -> None:
    global FAILED
    FAILED += 1
    log(f"FAIL {name}: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def timed_import(label: str, module: str, timeout_s: int = 90):
    log(f"import {label}  (fail if silent >{timeout_s}s)")
    t = time.time()
    try:
        import signal

        def _timeout(signum, frame):
            raise TimeoutError(f"{label} import exceeded {timeout_s}s")

        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(timeout_s)
        try:
            mod = __import__(module)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        ok(label, f"{getattr(mod, '__version__', '')}  {time.time() - t:.1f}s")
        return mod
    except Exception as exc:
        fail(label, exc)
        if "stable_baselines3" in module or "Timeout" in type(exc).__name__:
            log("Colab Python 3.13 often hangs here. Stop the cell and run locally with Python 3.12.")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Probe one off-screen frame")
    args = parser.parse_args()

    log(f"python {sys.version.split()[0]}  exe={sys.executable}")
    log(f"cwd={Path.cwd()}  MUJOCO_GL={GL}")
    log(f"repo={ROOT}  src_exists={(ROOT / 'src' / 'quad_loco').is_dir()}")

    if sys.version_info >= (3, 13):
        log("WARN Python 3.13: stable_baselines3 import may hang on Colab. Local 3.12 is the supported path.")

    timed_import("numpy", "numpy")
    timed_import("yaml", "yaml")
    timed_import("gymnasium", "gymnasium")
    timed_import("mujoco", "mujoco")
    timed_import("onnx", "onnx")
    timed_import("torch", "torch")
    timed_import("stable_baselines3", "stable_baselines3", timeout_s=90)

    t = time.time()
    try:
        from quad_loco.env import QuadrupedVelocityEnv
        from quad_loco.paths import scene_xml

        xml = scene_xml()
        if not xml.is_file():
            from quad_loco.convert_urdf import convert

            log("scene xml missing; converting URDF")
            xml = convert()
        ok("quad_loco.env", f"{time.time() - t:.1f}s  xml={xml}")
    except Exception as exc:
        fail("quad_loco.env", exc)
        return 1

    t = time.time()
    try:
        env = QuadrupedVelocityEnv(xml_path=xml, easy=True, command=(0.4, 0.0, 0.0))
        obs, info = env.reset(seed=0)
        assert obs.shape == (48,), obs.shape
        for _ in range(10):
            obs, rew, term, trunc, info = env.step(env.action_space.sample())
        ok(
            "env step",
            f"{time.time() - t:.1f}s  obs={obs.shape}  h={info.get('base_height'):.3f}  rew={rew:.2f}",
        )
    except Exception as exc:
        fail("env step", exc)
        return 1

    if args.render:
        t = time.time()
        log("render probe (first OSMesa/EGL frame can take 10-60s)")
        try:
            env.render_mode = "rgb_array"
            frame = env.render()
            dt = time.time() - t
            ok("render", f"{dt:.1f}s  shape={getattr(frame, 'shape', None)}")
        except Exception as exc:
            fail("render", exc)
            log("render is optional; eval --video will fail until OSMesa/EGL works")
    else:
        log("skip render (pass --render to probe video GL)")

    env.close()
    if FAILED:
        log(f"DONE with {FAILED} failure(s)")
        CLOCK.stop()
        return 1
    log("DONE all checks passed")
    CLOCK.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
