"""GGUF compatibility report.

llama.cpp commit 85286f3 registers ``Olmo3ForCausalLM`` on the OLMO2 GGUF
architecture and writes the sliding-window pattern. Public GGUF builds of
``allenai/Olmo-3-7B-Instruct`` exist (for example bartowski, built with
llama.cpp b7120). This module does not run that converter unless a path is
supplied and ``execute`` is true. A report with ``conversion_executed: false``
is not a successful export.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from myna_ai.models.registry import ModelSpec


def assess_gguf(spec: ModelSpec, *, converter: Path | None = None) -> dict[str, Any]:
    located = _locate_converter(converter)
    if spec.architecture == "olmo3":
        return {
            "model_key": spec.key,
            "hf_architecture": "Olmo3ForCausalLM",
            "gguf_architecture_label": "olmo2",
            "converter_support": "llama.cpp since b7120 (Olmo3ForCausalLM registered onto MODEL_ARCH.OLMO2)",
            "chat_format": "ChatML, eos <|endoftext|>",
            "converter_found": located is not None,
            "converter_path": None if located is None else str(located),
            "conversion_executed": False,
            "artifact_written": False,
            "supported": True,
            "claim": "Compatibility is documented from llama.cpp and public GGUF repos. This environment has not converted a checkpoint.",
            "alternative_runtime": "transformers AutoModelForCausalLM from a local directory, with the Python inference client",
        }
    if spec.architecture == "qwen2":
        return {
            "model_key": spec.key,
            "hf_architecture": spec.transformers_class,
            "gguf_architecture_label": "qwen2",
            "converter_support": "llama.cpp qwen2 support is widely available. Not executed here.",
            "converter_found": located is not None,
            "conversion_executed": False,
            "artifact_written": False,
            "supported": True,
            "licence_verified": spec.licence_verified,
            "claim": "Do not convert or redistribute until the Qwen card licence is re-checked.",
        }
    return {
        "model_key": spec.key,
        "supported": False,
        "conversion_executed": False,
        "artifact_written": False,
        "reason": f"{spec.architecture} is not an OLMo 3 or Qwen 2 GGUF target in this repository.",
        "alternative_runtime": "The Python transformers runtime for the smoke model.",
    }


def _locate_converter(explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit
    env = os.environ.get("LLAMA_CPP_CONVERT")
    if env and Path(env).is_file():
        return Path(env)
    found = shutil.which("convert_hf_to_gguf.py")
    if found:
        return Path(found)
    return None
