"""Exact duplicate detection for conversation records."""

from __future__ import annotations

from typing import Any

from myna_ai.common.hashing import sha256_json


def _fingerprint(record: dict[str, Any]) -> str:
    body = {
        "category": record.get("category"),
        "messages": record.get("messages"),
        "tools": record.get("tools"),
        "project_context": record.get("project_context"),
    }
    return sha256_json(body)


def drop_exact_duplicates(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep the first record for each fingerprint."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    for record in records:
        fingerprint = _fingerprint(record)
        record_id = str(record.get("id") or "")
        if fingerprint in seen:
            dropped.append({"id": record_id, "duplicate_of": seen[fingerprint], "fingerprint": fingerprint})
            continue
        seen[fingerprint] = record_id
        kept.append(record)
    return kept, dropped
