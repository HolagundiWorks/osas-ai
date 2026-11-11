"""Merge a local LoRA adapter into a full-precision model directory.

The base weights are loaded only from the checkpoint or from
``training_config.model_path``. This module does not contact the Hub.
QLoRA checkpoints are refused because merging a 4-bit adapter was not tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from myna_ai.common.errors import ExportError
from myna_ai.common.io import read_json, write_json
from myna_ai.export.manifest import build_manifest
from myna_ai.models.registry import require_model
from myna_ai.training.config import TrainingConfig
from myna_ai.training.tokenizer import CharTokenizer


def merge_adapter_checkpoint(
    checkpoint: Path,
    destination: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if destination.exists():
        raise ExportError(f"Export destination already exists: {destination}")
    config_path = checkpoint / "training_config.json"
    weights = checkpoint / "model.safetensors"
    if not config_path.is_file() or not weights.is_file():
        raise ExportError(
            "Merge needs training_config.json and model.safetensors in the checkpoint. "
            "Weights are not downloaded during export."
        )
    config = TrainingConfig.model_validate(read_json(config_path))
    if config.method == "qlora":
        raise ExportError(
            "QLoRA adapters are not merged here. Merging needs a full-precision base, "
            "and dequantising an untested 4-bit checkpoint is refused."
        )
    spec = require_model(config.model_key)
    if spec.local_constructor != "gpt2-random":
        model_path = config.model_path
        if model_path is None or not (model_path / "config.json").is_file():
            raise ExportError(
                "Merging this model requires training_config.model_path to be a local directory "
                "that already contains config.json. Export does not download the base weights."
            )
    tokenizer = _load_tokenizer(checkpoint, config, spec)
    from myna_ai.training.loop import _load_model
    from safetensors.torch import load_file

    model = _load_model(config, spec, tokenizer)
    missing, unexpected = model.load_state_dict(load_file(str(weights)), strict=False)
    if unexpected:
        raise ExportError(f"Checkpoint has unexpected tensors: {unexpected[:5]}")
    if missing:
        raise ExportError(
            f"Refusing to merge because {len(missing)} tensors were absent from the checkpoint, "
            f"including {missing[:3]}."
        )
    merged = model.merge_and_unload()
    destination.mkdir(parents=True)
    merged.save_pretrained(destination, safe_serialization=True)
    if isinstance(tokenizer, CharTokenizer):
        tokenizer.save(destination / "char_tokenizer.json")
    else:
        tokenizer.save_pretrained(destination)
    payload = {
        "artifact_type": "merged_full_precision",
        "model_key": spec.key,
        "base_model": spec.hf_id,
        "method": config.method,
        "promotion_allowed": False,
        "olmo_7b_trained": False,
        "smoke_test": spec.purpose == "smoke_test_only",
        "notes": "PEFT merge_and_unload wrote a full-precision directory. This is not a GGUF file.",
        **(metadata or {}),
    }
    write_json(destination / "export_metadata.json", payload)
    manifest = build_manifest(destination, payload)
    write_json(destination / "manifest.json", manifest)
    return manifest


def _load_tokenizer(checkpoint: Path, config: TrainingConfig, spec: Any) -> Any:
    if spec.local_constructor == "gpt2-random":
        path = checkpoint / "char_tokenizer.json"
        if path.is_file():
            return CharTokenizer.load(path)
        return CharTokenizer()
    from transformers import AutoTokenizer

    source = str(config.model_path or spec.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
