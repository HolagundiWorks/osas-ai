"""Plan a recovery fine-tune after pruning. This does not launch training."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
        "notes": [
            "Load the pruned checkpoint as model_path.",
            "Keep the original checkpoint for rollback.",
            "Run the same evaluation suite before any promotion.",
            "Do not delete the unpruned adapter.",
        ],
    }
