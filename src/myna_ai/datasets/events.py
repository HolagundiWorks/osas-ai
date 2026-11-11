"""Parse CAD Environments interaction traces without treating them as API calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Fields that must never be copied into a training prompt.
_SENSITIVE_KEYS = {"text", "content", "clipboard", "characters"}
_SUMMARY_KEYS = ("type", "timestamp", "appName", "bundleId", "name", "windowTitle", "button", "isDown")


def summarise_events(path: Path, *, max_lines: int | None = None) -> dict[str, Any]:
    """Count event types. Coordinates and clipboard payloads are discarded."""
    counts: dict[str, int] = {}
    applications: set[str] = set()
    parse_errors = 0
    seen = 0
    withheld = ["coordinates", "clipboard_payloads", "selected_text", "key_characters"]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if max_lines is not None and seen >= max_lines:
                break
            seen += 1
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if not isinstance(event, dict):
                parse_errors += 1
                continue
            event_type = str(event.get("type") or "unknown")
            counts[event_type] = counts.get(event_type, 0) + 1
            app = event.get("appName") or event.get("name")
            if isinstance(app, str) and app:
                applications.add(app)
            # Touching sensitive keys here only to record that they were dropped.
            for key in _SENSITIVE_KEYS:
                if key in event and event_type in {"clipboard", "text_selection", "key"}:
                    pass
    return {
        "format": "jsonl",
        "lines_read": seen,
        "parse_errors": parse_errors,
        "counts_by_type": dict(sorted(counts.items())),
        "applications": sorted(applications),
        "executable": False,
        "mapping_status": "unmapped_interaction_trace",
        "reason": "Desktop interaction traces are not MynaCAD or CoreOps API calls.",
        "withheld_fields": withheld,
        "retained_summary_fields": list(_SUMMARY_KEYS),
    }


def summarise_event_payload(payload: Any) -> dict[str, Any]:
    """Accept a JSON list, a JSONL string, or an already-summarised document."""
    if isinstance(payload, dict) and "counts_by_type" in payload:
        payload["executable"] = False
        return payload
    if isinstance(payload, list):
        counts: dict[str, int] = {}
        for event in payload:
            if isinstance(event, dict):
                event_type = str(event.get("type") or "unknown")
                counts[event_type] = counts.get(event_type, 0) + 1
        return {
            "format": "json-array",
            "lines_read": len(payload),
            "parse_errors": 0,
            "counts_by_type": dict(sorted(counts.items())),
            "executable": False,
            "mapping_status": "unmapped_interaction_trace",
            "reason": "Desktop interaction traces are not MynaCAD or CoreOps API calls.",
        }
    if isinstance(payload, str):
        counts = {}
        errors = 0
        lines = 0
        for line in payload.splitlines():
            if not line.strip():
                continue
            lines += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if isinstance(event, dict):
                event_type = str(event.get("type") or "unknown")
                counts[event_type] = counts.get(event_type, 0) + 1
        return {
            "format": "jsonl",
            "lines_read": lines,
            "parse_errors": errors,
            "counts_by_type": dict(sorted(counts.items())),
            "executable": False,
            "mapping_status": "unmapped_interaction_trace",
            "reason": "Desktop interaction traces are not MynaCAD or CoreOps API calls.",
        }
    return {
        "format": "unrecognised",
        "executable": False,
        "mapping_status": "unmapped_interaction_trace",
        "parse_errors": 1,
        "reason": "Event payload was not a JSONL trace or a JSON array.",
    }
