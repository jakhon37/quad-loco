from __future__ import annotations

import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from quad_loco.runtime import format_hms


def _fmt(value: float, spec: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return format(value, spec)


class CompactLogCallback(BaseCallback):
    """One-line status every N rollouts instead of SB3's full tables.

    Watch: rew up, len toward 1000, ev toward 1, kl around 0.01, ent not collapsing.
    TensorBoard still records the full SB3 set.
    """

    HEADER = (
        "runtime | steps | fps  rew  len | ev  kl  ent   "
        "# rew↑  len→1000  ev→1  kl~0.01  ent should not crash to 0"
    )

    def __init__(self, every: int = 10, total_timesteps: int = 0) -> None:
        super().__init__()
        self.every = max(1, every)
        self.total_timesteps = total_timesteps
        self._rollouts = 0
        self._header_printed = False

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        if not self._header_printed:
            print(self.HEADER, flush=True)
            self._header_printed = True

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if self._rollouts != 1 and self._rollouts % self.every != 0:
            return
        started = getattr(self.model, "start_time", None)
        if started:
            elapsed = max((time.time_ns() - started) / 1e9, 1e-8)
            fps = self.num_timesteps / elapsed
            wall = format_hms(elapsed)
        else:
            fps = float("nan")
            wall = "—"
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf:
            rew = float(np.mean([ep["r"] for ep in buf]))
            length = float(np.mean([ep["l"] for ep in buf]))
        else:
            rew = length = float("nan")
        # train/* is from the previous PPO update (recorded after this callback
        # on the first rollout).
        kv = self.logger.name_to_value
        ev = kv.get("train/explained_variance", float("nan"))
        kl = kv.get("train/approx_kl", float("nan"))
        ent = kv.get("train/entropy_loss", float("nan"))
        total = f"/{self.total_timesteps}" if self.total_timesteps else ""
        print(
            f"[runtime {wall}] {self.num_timesteps:>8}{total} | "
            f"fps={_fmt(fps, '.0f')}  rew={_fmt(rew, '.1f')}  len={_fmt(length, '.0f')} | "
            f"ev={_fmt(ev, '.2f')}  kl={_fmt(kl, '.3f')}  ent={_fmt(ent, '.2f')}",
            flush=True,
        )
