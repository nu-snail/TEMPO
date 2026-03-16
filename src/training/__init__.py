"""GRPO training pipeline for temporal leakage prevention."""

__all__ = ["TrainConfig", "run_training"]


def __getattr__(name: str):
    if name in ("TrainConfig", "run_training"):
        from .train import TrainConfig, run_training
        return {"TrainConfig": TrainConfig, "run_training": run_training}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
