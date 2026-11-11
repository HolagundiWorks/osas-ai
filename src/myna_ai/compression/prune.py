"""Structured MLP pruning for OLMo-style gate/up/down projections.

The pruner copies weights into a new module. The module passed in is not edited.
Per-layer intermediate sizes that differ from each other are rejected because
Olmo3Config stores one ``intermediate_size``. Attention tensors are not pruned.
A parameter reduction is not reported as a latency or memory reduction.
"""

from __future__ import annotations

from typing import Any

from myna_ai.common.errors import UnsupportedArchitectureError


def prune_mlp_module(module: Any, *, percentage: float, calibration: Any, alignment: int = 1) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from torch import nn

    _require_olmo_mlp(module)
    if percentage < 0 or percentage >= 1:
        raise UnsupportedArchitectureError("Pruning percentage must be in the range [0, 1).")
    if alignment < 1:
        raise UnsupportedArchitectureError("Alignment must be a positive integer.")
    with torch.no_grad():
        activated = F.silu(module.gate_proj(calibration)) * module.up_proj(calibration)
        reduce_dims = tuple(range(activated.ndim - 1))
        importance = activated.abs().mean(dim=reduce_dims)
    intermediate = int(importance.numel())
    if intermediate % alignment != 0:
        raise UnsupportedArchitectureError(
            f"Intermediate size {intermediate} is not divisible by alignment {alignment}."
        )
    drop = int(round(intermediate * percentage))
    drop = (drop // alignment) * alignment
    keep = intermediate - drop
    if keep < alignment:
        raise UnsupportedArchitectureError(
            f"Pruning {percentage:.0%} of {intermediate} units would leave fewer than {alignment} units."
        )
    index = torch.topk(importance, keep).indices.sort().values
    pruned = nn.Module()
    pruned.gate_proj = _sliced_linear(module.gate_proj, index, row=True)
    pruned.up_proj = _sliced_linear(module.up_proj, index, row=True)
    pruned.down_proj = _sliced_linear(module.down_proj, index, row=False)

    def forward(hidden):  # noqa: ANN001
        return pruned.down_proj(F.silu(pruned.gate_proj(hidden)) * pruned.up_proj(hidden))

    pruned.forward = forward  # type: ignore[method-assign]
    report = {
        "requested_percentage": percentage,
        "intermediate_before": intermediate,
        "intermediate_after": keep,
        "units_removed": intermediate - keep,
        "parameter_fraction_removed": (intermediate - keep) / intermediate,
        "alignment": alignment,
        "attention_pruned": False,
        "original_module_modified": False,
        "latency_claim": None,
        "memory_claim": None,
        "promoted": False,
        "promotion_reason": "A pruned module is not promoted without a full evaluation against the unpruned model.",
    }
    return pruned, report


def _require_olmo_mlp(module: Any) -> None:
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(module, name, None)
        weight = getattr(layer, "weight", None)
        if weight is None:
            raise UnsupportedArchitectureError(
                "Structured MLP pruning needs gate_proj, up_proj and down_proj Linear layers. "
                f"Missing {name}. GPT-2 Conv1D blocks are a different layout and are not pruned by this routine."
            )


def _sliced_linear(source: Any, index: Any, *, row: bool) -> Any:
    import torch
    from torch import nn

    weight = source.weight.detach()
    if row:
        sliced = weight.index_select(0, index)
        layer = nn.Linear(weight.shape[1], sliced.shape[0], bias=source.bias is not None)
        layer.weight = torch.nn.Parameter(sliced.clone())
    else:
        sliced = weight.index_select(1, index)
        layer = nn.Linear(sliced.shape[1], weight.shape[0], bias=source.bias is not None)
        layer.weight = torch.nn.Parameter(sliced.clone())
    if source.bias is not None:
        bias = source.bias.detach()
        if row:
            bias = bias.index_select(0, index)
        layer.bias = torch.nn.Parameter(bias.clone())
    return layer


def build_tiny_olmo_mlp(hidden: int = 8, intermediate: int = 16, seed: int = 0) -> Any:
    import torch
    from torch import nn

    generator = torch.Generator().manual_seed(seed)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            for layer in (self.gate_proj, self.up_proj, self.down_proj):
                torch.nn.init.normal_(layer.weight, generator=generator)

        def forward(self, hidden_states):  # noqa: ANN001
            import torch.nn.functional as F

            return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

    return Block()


def assess_olmo3_pruning() -> dict[str, Any]:
    """Architecture constraints for the verified OLMo 3 7B config. No weights are loaded."""
    return {
        "architecture": "olmo3",
        "intermediate_size": 11008,
        "hidden_size": 4096,
        "supported_strategy": "uniform_structured_mlp",
        "targets_percent": [0.05, 0.10, 0.15],
        "constraints": [
            "All layers share one intermediate_size, so every MLP must keep the same number of units.",
            "gate_proj, up_proj and down_proj must be sliced with the same index.",
            "Attention projections and the 32/32 head layout are left unchanged.",
            "11008 is not divisible by 64. Round the kept width to a divisor the runtime can load, or keep alignment at 1.",
            "llama.cpp GGUF conversion of a changed intermediate_size was not tested. Do not export a pruned model to GGUF from this repository.",
        ],
        "gguf_pruned_export": "unsupported_until_conversion_test",
        "in_place_prune": False,
    }
