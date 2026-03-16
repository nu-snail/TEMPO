"""Tinker RL environment and dataset for temporal leakage prevention.

Implements the Env / EnvGroupBuilder / RLDataset hierarchy required by
tinker-cookbook's RL training loop.  Each environment corresponds to a
single TaskInstance; groups of G environments share the same prompt so
GRPO can center rewards within the group.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    StepResult,
)
from tinker_cookbook.completers import StopCondition  # list[str] | list[int]

import tinker
from tinker import types as tinker_types

from src.shared.data_loader import TaskInstance
from src.shared.prompts.builder import build_prompt


# ---------------------------------------------------------------------------
# Single-turn prediction environment
# ---------------------------------------------------------------------------

class TemporalPredictionEnv(Env):
    """Single-turn environment: one prompt, one completion, episode done.

    The reward is computed externally by the training loop (not here)
    because it requires expensive API calls that benefit from batching.
    The step method returns reward=0.0; the real reward is injected via
    ``compute_group_rewards`` in the group builder.
    """

    def __init__(
        self,
        instance: TaskInstance,
        renderer: Any,
        tokenizer: Any,
        max_tokens: int = 4096,
    ) -> None:
        self.instance = instance
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._completion_text: str = ""

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        system, user = build_prompt(self.instance)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        model_input = self.renderer.build_generation_prompt(messages)
        stop: StopCondition = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action: Action) -> StepResult:
        self._completion_text = self.tokenizer.decode(action, skip_special_tokens=True)
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[],
        )

    @property
    def completion_text(self) -> str:
        return self._completion_text


# ---------------------------------------------------------------------------
# Group builder: G copies of the same prompt for GRPO
# ---------------------------------------------------------------------------

class TemporalGroupBuilder(EnvGroupBuilder):
    """Build G identical environments for one TaskInstance (GRPO group)."""

    def __init__(
        self,
        instance: TaskInstance,
        group_size: int,
        renderer: Any,
        tokenizer: Any,
        max_tokens: int = 4096,
    ) -> None:
        self.instance = instance
        self.group_size = group_size
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    async def make_envs(self) -> Sequence[Env]:
        return [
            TemporalPredictionEnv(
                instance=self.instance,
                renderer=self.renderer,
                tokenizer=self.tokenizer,
                max_tokens=self.max_tokens,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return [self.instance.task, "temporal"]


# ---------------------------------------------------------------------------
# Dataset: uniform task-mixed batches
# ---------------------------------------------------------------------------

@dataclass
class TemporalRLDataset(RLDataset):
    """Dataset that produces uniform 33/33/33 task-mixed batches.

    Each ``get_batch`` returns ``batch_size`` EnvGroupBuilders drawn
    uniformly across tasks.  The dataset is reshuffled every epoch.
    """
    instances: list[TaskInstance]
    batch_size: int
    group_size: int
    renderer: Any
    tokenizer: Any
    max_tokens: int = 4096
    seed: int = 42

    _order: list[int] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reshuffle()

    def _reshuffle(self) -> None:
        """Stratified shuffle: draw equally from each task."""
        by_task: dict[str, list[int]] = {}
        for i, inst in enumerate(self.instances):
            by_task.setdefault(inst.task, []).append(i)

        rng = random.Random(self.seed)
        for idxs in by_task.values():
            rng.shuffle(idxs)

        # Interleave tasks for uniform mixing
        pools = {t: iter(idxs) for t, idxs in by_task.items()}
        tasks = sorted(by_task.keys())
        order: list[int] = []
        exhausted: set[str] = set()
        while len(exhausted) < len(tasks):
            for t in tasks:
                if t in exhausted:
                    continue
                try:
                    order.append(next(pools[t]))
                except StopIteration:
                    exhausted.add(t)
        self._order = order

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        start = (index * self.batch_size) % len(self._order)
        indices = []
        for i in range(self.batch_size):
            idx = self._order[(start + i) % len(self._order)]
            indices.append(idx)

        return [
            TemporalGroupBuilder(
                instance=self.instances[idx],
                group_size=self.group_size,
                renderer=self.renderer,
                tokenizer=self.tokenizer,
                max_tokens=self.max_tokens,
            )
            for idx in indices
        ]

    def __len__(self) -> int:
        return max(1, len(self._order) // self.batch_size)
