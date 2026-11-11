"""CAD Environments adapter.

The public dataset ``markov-ai/cad-environments`` (dataset card and
``data/workflows.parquet`` at revision 78aea4f2) is an index of 51 human
desktop workflows plus large media files. It is not a set of neural-network
weights and it is not a MynaCAD command log.

Default ingestion reads the workflow index and, when present, ``task_desc.json``,
``narration.json`` and a summarised ``events.json``. Video, input files and
output CAD files are not downloaded by this adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from myna_ai.common.hashing import sha256_json
from myna_ai.datasets.events import summarise_event_payload, summarise_events
from myna_ai.datasets.narration import parse_narration, semantic_operations
from myna_ai.datasets.prompts import MYNA_SYSTEM_PROMPT

DATASET_ID = "markov-ai/cad-environments"
DATASET_REVISION = "78aea4f2cdde9d1ef2fc1780c12689e6ec06b1e9"
LICENCE = "cc-by-nc-4.0"
INDEX_COLUMNS = (
    "workflow_id",
    "software",
    "software_slug",
    "domain",
    "task",
    "description",
    "deliverables",
    "duration_hours",
    "platform",
    "event_count",
    "fps",
    "total_frames",
    "clip_url",
    "events_url",
    "frame_events_url",
    "narration_url",
    "task_description_url",
    "task_overview_url",
    "rubric_url",
    "input_files_url",
    "output_files_url",
)
DEFAULT_SOFTWARE_SLUGS = (
    "autocad",
    "sketchup",
    "revit-architecture",
    "revit-structure",
    "revit-mep",
    "staad-pro",
)
# Card totals. Sibling listings from the Hub API are truncated and are not a size source.
CARD_WORKFLOW_COUNT = 51
CARD_TOTAL_HOURS = 99.03
CARD_TOTAL_BYTES_REPORTED = 152 * 10**9
INDEX_PARQUET_BYTES = 40867


def category_for_slug(slug: str) -> str:
    if slug in {"revit-structure", "staad-pro"}:
        return "structural"
    if slug in {"revit-architecture", "revit-mep"}:
        return "bim"
    return "cad"


class CadEnvironmentsAdapter:
    name = "cad_environments"
    licence = LICENCE

    def inspect(self, source: Path) -> dict[str, Any]:
        rows = load_index(source)
        missing = [column for column in INDEX_COLUMNS if rows and column not in rows[0]]
        counts: dict[str, int] = {}
        for row in rows:
            slug = str(row.get("software_slug"))
            counts[slug] = counts.get(slug, 0) + 1
        return {
            "dataset_id": DATASET_ID,
            "revision_verified": DATASET_REVISION,
            "licence": LICENCE,
            "gated": False,
            "access": "public",
            "splits_in_card": ["train"],
            "index_rows": len(rows),
            "index_columns": list(rows[0].keys()) if rows else [],
            "missing_expected_columns": missing,
            "software_slug_counts": counts,
            "card_workflow_count": CARD_WORKFLOW_COUNT,
            "card_hours": CARD_TOTAL_HOURS,
            "card_total_file_size_bytes_reported": CARD_TOTAL_BYTES_REPORTED,
            "index_parquet_bytes_observed": INDEX_PARQUET_BYTES,
            "downloads_media": False,
            "notes": [
                "workflows.parquet is the Hub dataset-viewer index and is about 40 KB.",
                "The dataset card reports about 152 GB because of screen recordings and CAD files.",
                "This adapter does not download clips, input_files or output_files.",
                "events.json lines are desktop UI events, not Myna API calls.",
                "CC BY-NC 4.0 blocks commercial training unless a separate licence is obtained.",
            ],
        }

    def load_records(self, source: Path, options: dict[str, Any]) -> list[dict[str, Any]]:
        slugs = tuple(options.get("software_slugs") or DEFAULT_SOFTWARE_SLUGS)
        rows = [row for row in load_index(source) if str(row.get("software_slug")) in slugs]
        asset_root = Path(options["asset_root"]) if options.get("asset_root") else source.parent
        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(workflow_to_conversation(row, asset_root))
        return records


def load_index(source: Path) -> list[dict[str, Any]]:
    if source.is_dir():
        for name in ("workflows.json", "workflows.jsonl", "index.json", "workflows.parquet"):
            candidate = source / name
            if candidate.is_file():
                source = candidate
                break
        else:
            raise FileNotFoundError(f"No workflow index inside {source}")
    if source.suffix == ".parquet":
        return _read_parquet(source)
    if source.suffix == ".jsonl":
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "workflows" in payload:
        payload = payload["workflows"]
    if not isinstance(payload, list):
        raise ValueError(f"Workflow index must be a list: {source}")
    return payload


def workflow_to_conversation(row: dict[str, Any], asset_root: Path) -> dict[str, Any]:
    workflow_id = str(row["workflow_id"])
    slug = str(row.get("software_slug") or "")
    task_payload, narration_payload, event_summary, asset_notes = _load_assets(asset_root, slug, workflow_id)
    narration = parse_narration(narration_payload) if narration_payload is not None else {
        "summary": "",
        "annotations": [],
        "warnings": ["Narration file was not present. Only the workflow index was used."],
        "parsed": False,
    }
    operations = semantic_operations(narration, task_payload if isinstance(task_payload, dict) else None)
    user_text = _user_text(row, task_payload if isinstance(task_payload, dict) else None, narration, event_summary)
    assistant = {
        "role": "assistant",
        "content": _assistant_text(operations, event_summary),
    }
    record = {
        "id": f"cadenv-{workflow_id}",
        "category": category_for_slug(slug),
        "messages": [
            {"role": "system", "content": MYNA_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            assistant,
        ],
        "tools": [],
        "project_context": {
            "source_software": row.get("software"),
            "software_slug": slug,
            "domain": row.get("domain"),
            "api_mapping": "none",
        },
        "expected_validation": {
            "schema_valid": True,
            "state_changing": False,
            "requires_authorisation": False,
            "must_not_invent_quantities": True,
            "notes": "Semantic candidates only. No Myna tool call is generated from UI traces.",
        },
        "api_status": "example_schema",
        "source": {
            "dataset_id": DATASET_ID,
            "licence": LICENCE,
            "adapter": "cad_environments",
            "adapter_version": "1",
            "workflow_id": workflow_id,
            "revision": DATASET_REVISION,
            "notes": " ".join(asset_notes),
            "withheld_from_training": [
                "narration.annotations[].details",
                "events coordinates",
                "clipboard payloads",
                "video",
                "input_files",
                "output_files",
            ],
        },
    }
    record["source"]["content_sha256"] = sha256_json(
        {"workflow_id": workflow_id, "task": row.get("task"), "operations": operations}
    )
    return record


def _load_assets(asset_root: Path, slug: str, workflow_id: str) -> tuple[Any, Any, dict[str, Any] | None, list[str]]:
    notes: list[str] = []
    directory = _workflow_dir(asset_root, slug, workflow_id)
    task_payload = _read_optional_json(directory / "task_desc.json")
    narration_payload = _read_optional_json(directory / "narration.json")
    event_summary = None
    events_path = directory / "events.json"
    if not events_path.is_file():
        events_path = directory / "events.jsonl"
    if events_path.is_file():
        event_summary = summarise_events(events_path)
        notes.append("Event trace summarised by type. Raw events were not copied into the prompt.")
    else:
        notes.append("No local event trace. Remote event files were not downloaded.")
    if task_payload is None:
        notes.append("No local task_desc.json.")
    if narration_payload is None:
        notes.append("No local narration.json.")
    return task_payload, narration_payload, event_summary, notes


def _workflow_dir(asset_root: Path, slug: str, workflow_id: str) -> Path:
    nested = asset_root / slug / workflow_id
    if nested.is_dir():
        return nested
    flat = asset_root / workflow_id
    if flat.is_dir():
        return flat
    return nested


def _read_optional_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _user_text(
    row: dict[str, Any],
    task: dict[str, Any] | None,
    narration: dict[str, Any],
    event_summary: dict[str, Any] | None,
) -> str:
    lines = [
        f"Software recorded in the source workflow: {row.get('software')} ({row.get('software_slug')}).",
        f"Task: {row.get('task')}",
        f"Description: {row.get('description')}",
        "The source is a human desktop recording. Interaction traces are context, not MynaCAD commands.",
    ]
    if task and task.get("dimensions"):
        lines.append("Dimensions stated in the task file:")
        for dimension in task["dimensions"]:
            if isinstance(dimension, dict):
                lines.append(
                    f"- {dimension.get('id')}: {dimension.get('element')} = {dimension.get('measurement')}"
                )
    if narration.get("summary"):
        lines.append("Narration summary: " + str(narration["summary"]))
    if narration.get("annotations"):
        lines.append("Narrated steps (intent only):")
        for annotation in narration["annotations"]:
            intent = annotation.get("intent") or annotation.get("action")
            if intent:
                lines.append(f"- {intent}")
    if event_summary:
        lines.append(
            "UI event counts were recorded and withheld from the action list: "
            + json.dumps(event_summary.get("counts_by_type"), sort_keys=True)
        )
    lines.append("Explain the semantic operations. Do not invent quantities or execute the recording.")
    return "\n".join(lines)


def _assistant_text(operations: list[dict[str, Any]], event_summary: dict[str, Any] | None) -> str:
    lines = [
        "I will not execute the desktop recording. Mouse, keyboard and window events are not MynaCAD API calls.",
        "No quantity or structural design result is available from this recording, so none is stated.",
        "Unverified semantic candidates, with myna_api left null:",
        json.dumps(operations, ensure_ascii=False, indent=2),
    ]
    if event_summary:
        lines.append(
            f"The trace contained {event_summary.get('lines_read', 0)} UI events "
            "and was not translated into tool calls."
        )
    if not operations:
        lines.append("The available text does not identify a verifiable modelling operation.")
    return "\n".join(lines)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading workflows.parquet requires the dataset extra: pip install -e '.[dataset]'"
        ) from exc
    table = pq.read_table(path)
    columns = table.to_pydict()
    size = table.num_rows
    rows = []
    for index in range(size):
        rows.append({key: columns[key][index] for key in columns})
    return rows
