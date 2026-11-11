"""Training configuration. Values are validated before a run starts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from myna_ai.common.config import load_yaml
from myna_ai.common.errors import ConfigError

Method = Literal["lora", "qlora"]
Precision = Literal["fp32", "fp16", "bf16"]
Backend = Literal["native", "trl"]


class TrainingConfig(BaseModel):
    model_key: str
    train_path: Path
    validation_path: Path | None = None
    output_dir: Path
    method: Method = "qlora"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    epochs: float = 1.0
    max_steps: int | None = None
    precision: Precision = "bf16"
    gradient_checkpointing: bool = True
    seed: int = 42
    logging_steps: int = 1
    eval_steps: int = 20
    save_steps: int = 20
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    warmup_steps: int = 0
    allow_cpu_smoke: bool = False
    backend: Backend = "native"
    resume_from_checkpoint: Path | None = None
    model_path: Path | None = None
    local_files_only: bool = True
    smoke_n_embd: int = 64
    smoke_n_layer: int = 2
    smoke_n_head: int = 4

    @field_validator("max_seq_length", "lora_r", "per_device_batch_size", "gradient_accumulation_steps")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("training sizes must be >= 1")
        return value

    @field_validator("learning_rate")
    @classmethod
    def _lr(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("learning_rate must be positive")
        return value

    @field_validator("epochs")
    @classmethod
    def _epochs(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("epochs must be positive")
        return value


def load_training_config(path: Path, overrides: dict | None = None) -> TrainingConfig:
    raw = load_yaml(path)
    if overrides:
        raw.update({key: value for key, value in overrides.items() if value is not None})
    try:
        return TrainingConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(str(exc)) from exc


def learning_rate_at(step: int, base: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base
    if step < warmup_steps:
        return base * float(step + 1) / float(warmup_steps)
    return base


def should_early_stop(eval_losses: list[float], patience: int, min_delta: float) -> bool:
    """Stop when the best loss has not improved for ``patience`` evaluations."""
    if patience <= 0 or len(eval_losses) <= patience:
        return False
    best = eval_losses[0]
    best_index = 0
    for index, loss in enumerate(eval_losses):
        if loss < best - min_delta:
            best = loss
            best_index = index
    return len(eval_losses) - 1 - best_index >= patience
