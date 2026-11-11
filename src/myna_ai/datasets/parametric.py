"""Optional adapter for local parametric CAD instruction files.

This does not download a public parametric dataset. It accepts a JSONL file
with instruction, optional context, response and category fields so a later
corpus can be added without a new pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from myna_ai.datasets.prompts import MYNA_SYSTEM_PROMPT

CATEGORIES = {"cad", "structural", "coreops", "bim", "safety"}


class ParametricInstructionAdapter:
    name = "parametric_instructions"
    licence = "local-unspecified"

    def inspect(self, source: Path) -> dict[str, Any]:
        rows = _read(source)
        return {
            "adapter": self.name,
            "rows": len(rows),
            "licence": self.licence,
            "notes": [
                "Local instruction files only.",
                "No external parametric CAD dataset is configured.",
                "Set source.licence on each row when the upstream licence is known.",
            ],
        }

    def load_records(self, source: Path, options: dict[str, Any]) -> list[dict[str, Any]]:
        licence = str(options.get("licence") or "local-unspecified")
        records = []
        for index, row in enumerate(_read(source)):
            category = str(row.get("category") or "cad")
            if category not in CATEGORIES:
                category = "cad"
            instruction = str(row["instruction"])
            context = row.get("context")
            response = str(row["response"])
            user = instruction if not context else instruction + "\nContext:\n" + json.dumps(context, ensure_ascii=False)
            records.append(
                {
                    "id": str(row.get("id") or f"parametric-{index + 1}"),
                    "category": category,
                    "messages": [
                        {"role": "system", "content": MYNA_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": response},
                    ],
                    "tools": row.get("tools") or [],
                    "project_context": row.get("project_context") or {},
                    "expected_validation": {
                        "schema_valid": True,
                        "state_changing": False,
                        "requires_authorisation": False,
                        "must_not_invent_quantities": True,
                    },
                    "api_status": "example_schema",
                    "source": {
                        "dataset_id": str(row.get("dataset_id") or "local-parametric-instructions"),
                        "licence": str(row.get("licence") or licence),
                        "adapter": self.name,
                        "adapter_version": "1",
                        "workflow_id": row.get("workflow_id"),
                        "notes": "Local parametric instruction row. Not a verified Myna API trace.",
                    },
                }
            )
        return records


def _read(source: Path) -> list[dict[str, Any]]:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if "instruction" not in payload or "response" not in payload:
                raise ValueError("Parametric rows require instruction and response.")
            rows.append(payload)
    return rows
