"""Configurable model registry. Training code reads this instead of hardcoding ids."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from myna_ai.common.config import load_yaml, repo_root
from myna_ai.common.errors import ConfigError

Purpose = Literal["production_candidate", "alternative", "smoke_test_only"]


class ModelSpec(BaseModel):
    """One loadable model entry."""

    key: str
    hf_id: str | None = None
    architecture: str
    hf_model_type: str
    transformers_class: str
    licence: str
    licence_verified: bool = False
    licence_notes: str = ""
    chat_template: str
    eos_token: str = "<|endoftext|>"
    pad_token: str | None = None
    lora_target_modules: list[str]
    fan_in_fan_out: bool = False
    qlora_supported: bool
    gguf_status: str
    min_transformers_version: str = "4.57.1"
    context_length: int
    default_training_seq_length: int
    min_vram_gb_qlora: float | None = None
    min_vram_gb_lora: float | None = None
    requires_gpu_for_training: bool = True
    allow_cpu_smoke: bool = False
    local_constructor: str | None = None
    purpose: Purpose
    parameter_note: str = ""
    verified_fields: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("lora_target_modules")
    @classmethod
    def _modules(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("lora_target_modules must not be empty")
        return value


class ModelRegistry(BaseModel):
    default_model: str
    models: dict[str, ModelSpec]

    def get(self, key: str) -> ModelSpec:
        try:
            return self.models[key]
        except KeyError as exc:
            known = ", ".join(sorted(self.models))
            raise ConfigError(f"Unknown model key '{key}'. Known keys: {known}") from exc


def registry_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return repo_root() / "configs" / "models" / "registry.yaml"


def load_registry(path: Path | None = None) -> ModelRegistry:
    raw = load_yaml(registry_path(path))
    models_raw = raw.get("models")
    if not isinstance(models_raw, dict) or not models_raw:
        raise ConfigError("Model registry must contain a non-empty 'models' mapping.")
    models: dict[str, ModelSpec] = {}
    for key, value in models_raw.items():
        if not isinstance(value, dict):
            raise ConfigError(f"Model entry '{key}' must be a mapping.")
        payload: dict[str, Any] = {"key": key, **value}
        models[key] = ModelSpec.model_validate(payload)
    default = raw.get("default_model")
    if default not in models:
        raise ConfigError("default_model must name an entry in models.")
    return ModelRegistry(default_model=str(default), models=models)


def require_model(key: str | None = None, path: Path | None = None) -> ModelSpec:
    registry = load_registry(path)
    return registry.get(key or registry.default_model)
