"""Deployment quantisation experiments.

These routines are not QLoRA. QLoRA is a training-time 4-bit load
(``BitsAndBytesConfig`` with NF4). The functions here measure a separate
deployment-time dynamic int8 experiment on ``nn.Linear`` modules.
"""

from __future__ import annotations

import io
from typing import Any


def describe_quantisation_modes() -> dict[str, str]:
    return {
        "qlora_training": "bitsandbytes NF4 load used only while training an adapter. It is not a portable runtime file.",
        "dynamic_int8_experiment": "torch dynamic quantisation of Linear layers for a local experiment. Not a GGUF file.",
        "gguf_deployment": "llama.cpp GGUF quantisation of a merged OLMo 3 checkpoint. Supported by llama.cpp since b7120, not executed in this repository.",
        "safetensors_fp32_or_bf16": "Full-precision or bf16 weights via transformers save_pretrained. This is the portable artifact this repository can actually write.",
    }


def dynamic_int8_size_report(module: Any) -> dict[str, Any]:
    """Quantise a copy-like experiment and compare serialised sizes.

    ``quantize_dynamic`` returns a new module. The caller should pass a module
    it can discard. The report does not claim a change in inference latency.
    """
    import torch
    from torch import nn

    quantize = getattr(torch.ao.quantization, "quantize_dynamic", None)
    if quantize is None:
        quantize = torch.quantization.quantize_dynamic
    quantised = quantize(module, {nn.Linear}, dtype=torch.qint8)
    original_bytes = _torch_bytes(module)
    quantised_bytes = _torch_bytes(quantised)
    return {
        "mode": "dynamic_int8_experiment",
        "implementation": "torch.ao.quantization.quantize_dynamic",
        "deprecated_api": True,
        "original_bytes": original_bytes,
        "quantised_bytes": quantised_bytes,
        "byte_ratio": quantised_bytes / original_bytes if original_bytes else None,
        "latency_measured": False,
        "memory_claim": None,
        "gguf": False,
        "qlora": False,
        "notes": (
            "torch.ao.quantization.quantize_dynamic emitted a deprecation warning on torch 2.15. "
            "This report is an experiment, not a deployment artifact and not a QLoRA or GGUF file."
        ),
    }


def _torch_bytes(module: Any) -> int:
    import torch

    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.tell()
