"""Lightweight progress bar for training and evaluation loops."""

from __future__ import annotations

import sys
import time


class ProgressBar:
    """Single-line progress bar with ETA and custom metrics.

    Usage::

        pb = ProgressBar(total=100, prefix="Train")
        for i in range(100):
            pb.update(i + 1, step_time=12.5, R="0.65", OLR="0.21")
        pb.finish()
    """

    BAR_WIDTH = 20

    def __init__(self, total: int, prefix: str = "") -> None:
        self.total = max(total, 1)
        self.prefix = prefix
        self._t0 = time.time()
        self._cumulative_step_time = 0.0

    def update(self, current: int, *, step_time: float = 0.0, **kv: str | float) -> None:
        self._cumulative_step_time += step_time
        # Use cumulative step time for ETA (excludes validation overhead)
        train_elapsed = self._cumulative_step_time if self._cumulative_step_time > 0 else (time.time() - self._t0)
        frac = current / self.total
        filled = int(self.BAR_WIDTH * frac)
        bar = "█" * filled + "░" * (self.BAR_WIDTH - filled)

        per_item = train_elapsed / current if current > 0 else 0
        remaining = per_item * (self.total - current)
        eta = _fmt_time(remaining)
        elapsed_str = _fmt_time(time.time() - self._t0)

        parts = [
            f"\r{self.prefix} [{current}/{self.total}]",
            f"{bar} {frac:5.1%}",
        ]
        if kv:
            metrics = " ".join(f"{k}={_fmt_val(v)}" for k, v in kv.items())
            parts.append(f"| {metrics}")
        parts.append(f"| {elapsed_str}<{eta}")

        line = " ".join(parts)
        sys.stderr.write(f"\r{line}\033[K")
        sys.stderr.flush()

    def finish(self, **kv: str | float) -> None:
        self.update(self.total, **kv)
        sys.stderr.write("\n")
        sys.stderr.flush()


def _fmt_time(secs: float) -> str:
    if secs < 0:
        return "??:??"
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m:02d}m{s:02d}s"


def _fmt_val(v: str | float) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
