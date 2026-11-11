"""Copy a local full-precision checkpoint without optimizer state."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from myna_ai.common.errors import ExportError
from myna_ai.common.io import write_json
from myna_ai.export.manifest import build_manifest

_ALLOWED_NAMES = {
    "config.json",
    "generation_config.json",
    "training_config.json",
    "trainer_state.json",
    "char_tokenizer.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
}


def export_full_precision(source: Path, destination: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Copy deployment weights and sidecars. Optimizer state is excluded.

    A LoRA training checkpoint's ``model.safetensors`` still contains adapter
    tensors. Use ``merge_adapter_checkpoint`` when the artifact must be a
    plain base model.
    """
    if not source.is_dir():
        raise ExportError(f"Full-precision source is not a directory: {source}")
    if destination.exists():
        raise ExportError(f"Export destination already exists: {destination}")
    selected = _select_files(source)
    if not selected:
        raise ExportError(
            f"No model safetensors file under {source}. "
            "Refusing to write a full-precision manifest without weights."
        )
    destination.mkdir(parents=True)
    for path in selected:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    payload = {
        "artifact_type": "full_precision",
        "promotion_allowed": False,
        "optimizer_included": False,
        "notes": (
            "Copied local safetensors only. Optimizer state was left behind. "
            "A LoRA checkpoint still contains adapter tensors until it is merged."
        ),
        **metadata,
    }
    write_json(destination / "export_metadata.json", payload)
    manifest = build_manifest(destination, payload)
    write_json(destination / "manifest.json", manifest)
    return manifest


def _select_files(source: Path) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if path.name == "optimizer.pt" or path.name == "manifest.json":
            continue
        if path.name in _ALLOWED_NAMES or _is_weight_file(path.name):
            selected.append(path)
    weights = [path for path in selected if _is_weight_file(path.name)]
    if not weights:
        return []
    return selected


def _is_weight_file(name: str) -> bool:
    return name.endswith(".safetensors") and name.startswith("model")
