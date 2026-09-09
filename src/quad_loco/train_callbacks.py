from __future__ import annotations

import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CompactLogCallback(BaseCallback):
    """One-line status every N rollouts instead of SB3's full tables."""

    def __init__(self, every: int = 10, total_timesteps: int = 0) -> None:
        super().__init__()
        self.every = max(1, every)
        self.total_timesteps = total_timesteps
        self._rollouts = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if self._rollouts != 1 and self._rollouts % self.every != 0:
            return
        started = getattr(self.model, "start_time", None)
        if started:
            elapsed = max((time.time_ns() - started) / 1e9, 1e-8)
            fps = self.num_timesteps / elapsed
        else:
            fps = float("nan")
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf:
            rew = float(np.mean([ep["r"] for ep in buf]))
            length = float(np.mean([ep["l"] for ep in buf]))
        else:
            rew = length = float("nan")
        total = f"/{self.total_timesteps}" if self.total_timesteps else ""
        print(
            f"{self.num_timesteps:>8}{total} steps | "
            f"fps={fps:.0f}  rew={rew:.1f}  len={length:.0f}",
            flush=True,
        )
