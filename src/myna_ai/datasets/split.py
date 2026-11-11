"""Workflow-level splits so examples from one workflow stay together."""

from __future__ import annotations

import hashlib
from typing import Any

from myna_ai.common.errors import ConfigError


def assign_split(key: str, *, seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
    if len(ratios) != 3:
        raise ConfigError("Split ratios must contain train, validation and test values.")
    if any(part < 0 for part in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ConfigError(f"Split ratios must be non-negative and sum to 1. Got {ratios}.")
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    train, validation, _test = ratios
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def split_key(record: dict[str, Any]) -> str:
    source = record.get("source") or {}
    workflow_id = source.get("workflow_id")
    if isinstance(workflow_id, str) and workflow_id:
        return workflow_id
    return str(record.get("id"))


def apply_splits(
    records: list[dict[str, Any]],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    keys = {split_key(record) for record in records}
    assignment = {key: assign_split(key, seed=seed, ratios=ratios) for key in keys}
    for record in records:
        split = assignment[split_key(record)]
        copied = dict(record)
        copied["split"] = split
        grouped[split].append(copied)
    return grouped
