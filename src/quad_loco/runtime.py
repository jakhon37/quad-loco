"""Wall-clock runtime prefix and an optional heartbeat while work is blocking."""

from __future__ import annotations

import threading
import time


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class RuntimeClock:
    def __init__(self, heartbeat_s: float = 15.0) -> None:
        self.t0 = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if heartbeat_s and heartbeat_s > 0:
            self._thread = threading.Thread(
                target=self._beat,
                args=(heartbeat_s,),
                name="runtime-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def stamp(self) -> str:
        return format_hms(self.elapsed())

    def log(self, msg: str) -> None:
        print(f"[runtime {self.stamp()}] {msg}", flush=True)

    def _beat(self, interval: float) -> None:
        while not self._stop.wait(interval):
            print(f"[runtime {self.stamp()}] still running …", flush=True)

    def stop(self) -> None:
        self._stop.set()
