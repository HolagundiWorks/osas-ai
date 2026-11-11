"""Inspect a local dataset mirror or the public Hugging Face dataset card."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from myna_ai.common.logging import redact
from myna_ai.datasets.cad_environments import (
    CARD_TOTAL_BYTES_REPORTED,
    CARD_TOTAL_HOURS,
    CARD_WORKFLOW_COUNT,
    DATASET_ID,
    DATASET_REVISION,
    INDEX_COLUMNS,
    INDEX_PARQUET_BYTES,
    LICENCE,
    CadEnvironmentsAdapter,
)

HF_API = "https://huggingface.co/api/datasets/{dataset_id}"


def inspect_local(source: Path) -> dict[str, Any]:
    return CadEnvironmentsAdapter().inspect(source)


def inspect_huggingface(dataset_id: str = DATASET_ID, *, timeout_s: float = 30) -> dict[str, Any]:
    """Read Hub metadata only. This does not download workflow media."""
    request = urllib.request.Request(HF_API.format(dataset_id=dataset_id))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    card = payload.get("cardData") or {}
    siblings = payload.get("siblings") or []
    return {
        "dataset_id": payload.get("id", dataset_id),
        "sha": payload.get("sha"),
        "private": payload.get("private"),
        "gated": payload.get("gated"),
        "licence_in_card": (card.get("license") if isinstance(card, dict) else None),
        "splits": _splits(card),
        "sibling_entries_returned": len(siblings),
        "sibling_list_truncated": len(siblings) >= 100000,
        "expected_index_columns": list(INDEX_COLUMNS),
        "known_revision_inspected_for_this_repo": DATASET_REVISION,
        "known_index_parquet_bytes": INDEX_PARQUET_BYTES,
        "card_workflow_count": CARD_WORKFLOW_COUNT,
        "card_hours": CARD_TOTAL_HOURS,
        "card_total_file_size_bytes_reported": CARD_TOTAL_BYTES_REPORTED,
        "downloads_media": False,
        "notes": [
            "Unauthenticated Hub responses include an HF warning. No token is written by this function.",
            redact(f"gated={payload.get('gated')} private={payload.get('private')}"),
        ],
    }


def _splits(card: Any) -> list[str]:
    if not isinstance(card, dict):
        return []
    found: list[str] = []
    for config in card.get("configs") or []:
        if not isinstance(config, dict):
            continue
        for item in config.get("data_files") or []:
            if isinstance(item, dict) and item.get("split"):
                found.append(str(item["split"]))
    return found
