"""Recovery after structured pruning.

``recover_pruned_mlp`` trains a copy of a pruned MLP so its outputs move
closer to the unpruned module. The original module is left unchanged.

``launch_recovery_finetune`` can resume the language-model trainer from a
pruned checkpoint. It does not start a job unless ``execute`` is true, and
the trainer still refuses OLMo 7B on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from myna_ai.common.errors import ExportError


def plan_recovery_finetune(
    *,
    pruned_checkpoint: Path,
    train_config: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "action": "recovery_finetune_plan",
        "launched": False,
        "pruned_checkpoint": str(pruned_checkpoint),
        "train_config": str(train_config),
        "output_dir": str(output_dir),
        "promoted": False,
        "olmo_7b_trained": False,
        "notes": [
            "Load the pruned checkpoint as the resume point.",
            "Keep the original checkpoint for rollback.",
            "Run the same evaluation suite before any promotion.",
            "Do not delete the unpruned adapter.",
            "Pass execute=True to start training. A plan does not launch a job.",
        ],
    }


def launch_recovery_finetune(
    *,
    pruned_checkpoint: Path,
    train_config: Path,
    output_dir: Path | None = None,
    execute: bool = False,
    max_steps: int | None = None,
) -> dict[str, Any]:
    destination = output_dir or Path("outputs/recovery")
    plan = plan_recovery_finetune(
        pruned_checkpoint=pruned_checkpoint,
        train_config=train_config,
        output_dir=destination,
    )
    if not execute:
        return plan
    if not pruned_checkpoint.exists():
        raise ExportError(f"Pruned checkpoint does not exist: {pruned_checkpoint}")
    if not train_config.is_file():
        raise ExportError(f"Training config does not exist: {train_config}")
    from myna_ai.common.io import write_json
    from myna_ai.training.config import load_training_config
    from myna_ai.training.loop import run_training

    overrides: dict[str, Any] = {"resume_from_checkpoint": pruned_checkpoint}
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if max_steps is not None:
        overrides["max_steps"] = max_steps
    config = load_training_config(train_config, overrides)
    result = run_training(config)
    plan["launched"] = True
    plan["promoted"] = False
    plan["olmo_7b_trained"] = bool(result.get("olmo_7b_trained"))
    plan["train_result"] = {
        "model_key": result.get("model_key"),
        "steps_completed": result.get("steps_completed"),
        "steps_started_from": result.get("steps_started_from"),
        "checkpoint": result.get("checkpoint"),
        "smoke_test": result.get("smoke_test"),
        "olmo_7b_trained": result.get("olmo_7b_trained"),
    }
    write_json(config.output_dir / "recovery_result.json", plan)
    return plan


def recover_pruned_mlp(
    original: Any,
    pruned: Any,
    calibration: Any,
    *,
    steps: int = 30,
    learning_rate: float = 1e-2,
) -> tuple[Any, dict[str, Any]]:
    """Fine-tune a copy of ``pruned`` toward ``original`` outputs.

    ``original`` is not modified. The returned module is not promoted.
    """
    import copy

    import torch

    if steps < 1:
        raise ExportError("Recovery steps must be at least 1.")
    if learning_rate <= 0:
        raise ExportError("Recovery learning rate must be positive.")
    original_state = {name: value.detach().clone() for name, value in original.state_dict().items()}
    recovered = copy.deepcopy(pruned)
    recovered.train()
    optimizer = torch.optim.AdamW(recovered.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            target = original(calibration)
        prediction = recovered(calibration)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise ExportError("Recovery loss was not finite. Training of the pruned copy stopped.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    for name, value in original.state_dict().items():
        if not torch.equal(value, original_state[name]):
            raise ExportError(f"Recovery modified the original module parameter {name}.")
    report = {
        "action": "mlp_recovery_finetune",
        "launched": True,
        "steps": steps,
        "learning_rate": learning_rate,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "original_module_modified": False,
        "promoted": False,
        "olmo_7b_trained": False,
        "promotion_reason": "A recovered MLP block is not a promotable language-model artifact.",
    }
    return recovered, report
