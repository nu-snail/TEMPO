"""Pure cosine warmup lambda schedule for multi-objective reward weighting.

    lambda(t) = lambda_min + (lambda_max - lambda_min) / 2 * (1 - cos(pi * t / T))

Smoothly ramps from lambda_min to lambda_max over T training steps,
ensuring the model focuses on leakage reduction early and gradually
increases performance weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LambdaSchedule:
    """Cosine warmup lambda schedule.

    Args:
        total_steps: Total training steps T.
        lambda_min: Performance weight at the start of training.
        lambda_max: Performance weight at the end of training.
    """
    total_steps: int
    lambda_min: float = 0.1
    lambda_max: float = 0.3

    # Kept for config compatibility but unused.
    theta_high: float = 0.3
    theta_low: float = 0.1
    ema_alpha: float = 0.1

    def get_lambda(self, step: int, val_olr: float | None = None) -> float:
        """Compute lambda(t) for the given step.

        ``val_olr`` is accepted for API compatibility but ignored.
        """
        t = min(step, self.total_steps)
        progress = t / max(self.total_steps, 1)
        return self.lambda_min + (self.lambda_max - self.lambda_min) / 2.0 * (
            1.0 - math.cos(math.pi * progress)
        )
