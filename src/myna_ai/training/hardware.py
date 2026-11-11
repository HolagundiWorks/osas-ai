"""Device checks. Full-size training never falls through to CPU."""

from __future__ import annotations

from typing import Any

from myna_ai.common.errors import GpuRequiredError, UnsupportedArchitectureError
from myna_ai.models.registry import ModelSpec
from myna_ai.training.config import TrainingConfig


def cuda_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "free_gb": None, "reason": "PyTorch is not installed."}
    if not torch.cuda.is_available():
        return {"available": False, "free_gb": None, "reason": "torch.cuda.is_available() is false."}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "free_gb": free / (1024**3),
        "total_gb": total / (1024**3),
        "device_name": torch.cuda.get_device_name(0),
        "reason": "",
    }


def assert_training_device(spec: ModelSpec, config: TrainingConfig, status: dict[str, Any] | None = None) -> dict[str, Any]:
    """Refuse CPU or undersized GPUs for OLMo-scale training."""
    status = cuda_status() if status is None else status
    if config.method == "qlora" and not spec.qlora_supported:
        raise UnsupportedArchitectureError(
            f"{spec.key} is not marked QLoRA-compatible. "
            "Use method=lora for the smoke model. QLoRA is the default only for the 7B registry entry."
        )
    needs_gpu = spec.requires_gpu_for_training or config.method == "qlora"
    smoke_on_cpu = spec.allow_cpu_smoke and config.allow_cpu_smoke and config.method == "lora"
    if needs_gpu and not smoke_on_cpu:
        if not status.get("available"):
            raise GpuRequiredError(
                f"Refusing to train {spec.key} with method={config.method} because no CUDA GPU is available "
                f"({status.get('reason')}). "
                f"QLoRA for OLMo 3 7B needs an NVIDIA GPU with at least {spec.min_vram_gb_qlora} GB free. "
                "A 24 GB card is the practical default at sequence length 2048. "
                "This process will not silently continue on CPU. "
                "Pipeline checks can use model_key=smoke-tiny, method=lora and allow_cpu_smoke=true."
            )
        required = spec.min_vram_gb_qlora if config.method == "qlora" else spec.min_vram_gb_lora
        free = status.get("free_gb")
        if required is not None and free is not None and free + 0.25 < required:
            raise GpuRequiredError(
                f"GPU '{status.get('device_name')}' has {free:.1f} GB free. "
                f"{spec.key} {config.method} is configured to require {required} GB. "
                "Lower the sequence length only through an explicit config change, or use a larger GPU. "
                "Training was not started."
            )
    if config.precision == "bf16" and not status.get("available") and not smoke_on_cpu:
        raise GpuRequiredError("bf16 training was requested without a CUDA device.")
    if config.precision != "fp32" and smoke_on_cpu:
        raise GpuRequiredError(
            "The CPU smoke path supports fp32 only. It does not silently cast a bf16 or fp16 config."
        )
    return status
