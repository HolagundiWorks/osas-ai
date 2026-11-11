"""Parse narration files from CAD Environments without inventing API calls."""

from __future__ import annotations

from typing import Any


def parse_narration(payload: Any) -> dict[str, Any]:
    """Normalise the narration shapes observed in the public dataset.

    Observed object shape (revision 78aea4f2, one STAAD.Pro workflow):
    ``{"annotations": [{"start_s", "end_s", "action", "intent", "app", "details"}], "summary"}``.
    A bare list of annotation objects is also accepted. Anything else is returned
    as an unparsed payload with a warning.
    """
    warnings: list[str] = []
    summary = ""
    annotations: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        summary = str(payload.get("summary") or "")
        source_rows = payload["annotations"]
    elif isinstance(payload, list):
        source_rows = payload
        warnings.append("Narration was a list rather than an annotations object.")
    else:
        return {
            "summary": "",
            "annotations": [],
            "warnings": ["Narration payload did not match the known object or list shape."],
            "parsed": False,
        }
    for row in source_rows:
        if not isinstance(row, dict):
            warnings.append("Skipped a non-object narration annotation.")
            continue
        annotations.append(
            {
                "start_s": row.get("start_s"),
                "end_s": row.get("end_s"),
                "action": _text(row.get("action")),
                "intent": _text(row.get("intent")),
                "app": _text(row.get("app")),
                # UI coordinates often live in details. Keep them out of the model prompt.
                "details_withheld": "details" in row,
            }
        )
    return {
        "summary": summary,
        "annotations": annotations,
        "warnings": warnings,
        "parsed": True,
    }


def semantic_operations(narration: dict[str, Any], task: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Turn narration into labelled semantic candidates.

    These records are not MynaCAD commands. ``myna_api`` is always null.
    """
    operations: list[dict[str, Any]] = []
    for index, annotation in enumerate(narration.get("annotations") or []):
        operations.append(
            {
                "operation_id": f"narration-{index + 1}",
                "source": "narration.annotations",
                "intent": annotation.get("intent") or "",
                "action_text": annotation.get("action") or "",
                "application": annotation.get("app") or "",
                "start_s": annotation.get("start_s"),
                "end_s": annotation.get("end_s"),
                "myna_api": None,
                "status": "unverified_semantic_candidate",
                "executable": False,
                "notes": "Human narration. Not an executable MynaCAD or CoreOps command.",
            }
        )
    for index, dimension in enumerate((task or {}).get("dimensions") or []):
        if not isinstance(dimension, dict):
            continue
        measurement = dimension.get("measurement")
        operations.append(
            {
                "operation_id": f"dimension-{dimension.get('id') or index + 1}",
                "source": "task_desc.dimensions",
                "element": dimension.get("element"),
                "measurement_text": measurement,
                "myna_api": None,
                "status": "unverified_semantic_candidate",
                "executable": False,
                "notes": "Dimension text copied from the task statement. Not a calculated result.",
            }
        )
    return operations


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
