"""Four-way comparison of evaluation reports.

Slot A is the original checkpoint, B the fine-tuned adapter, C a quantised
artifact and D a pruned artifact. A slot with no report is recorded as
unmeasured. Missing numbers are not filled in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from myna_ai.common.errors import ConfigError
from myna_ai.common.io import read_json, write_json

SLOTS = ("A", "B", "C", "D")
SLOT_ROLES = {
    "A": "original",
    "B": "finetuned",
    "C": "quantised",
    "D": "pruned",
}


def build_comparison_matrix(reports: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    unknown = sorted(set(reports) - set(SLOTS))
    if unknown:
        raise ConfigError(f"Unknown evaluation slots {unknown}. Use A, B, C or D.")
    slots: dict[str, Any] = {}
    for slot in SLOTS:
        report = reports.get(slot)
        if not report:
            slots[slot] = {
                "role": SLOT_ROLES[slot],
                "measured": False,
                "model_label": None,
                "promote": None,
                "metrics": None,
                "artifact_bytes": None,
                "olmo_7b_evaluated": False,
                "reason": f"No report was supplied for the {SLOT_ROLES[slot]} artifact.",
            }
            continue
        slots[slot] = {
            "role": SLOT_ROLES[slot],
            "measured": True,
            "model_label": report.get("model_label"),
            "promote": bool(report.get("promotion", {}).get("promote")),
            "metrics": report.get("metrics"),
            "artifact_bytes": report.get("artifact_bytes"),
            "olmo_7b_evaluated": bool(report.get("olmo_7b_evaluated")),
            "reason": None,
        }
    return {
        "format_version": "1",
        "olmo_7b_baseline_measured": bool(slots["A"]["measured"] and slots["A"]["olmo_7b_evaluated"]),
        "slots": slots,
    }


def load_report_slot(spec: str) -> tuple[str, dict[str, Any]]:
    slot, separator, raw_path = spec.partition("=")
    if separator != "=" or slot not in SLOTS:
        raise ConfigError(f"Expected a report as SLOT=directory, for example B=outputs/eval. Got {spec!r}.")
    path = Path(raw_path)
    report_path = path / "report.json" if path.is_dir() else path
    if not report_path.is_file():
        raise ConfigError(f"No evaluation report at {report_path}.")
    report = read_json(report_path)
    if not isinstance(report, dict) or "metrics" not in report:
        raise ConfigError(f"{report_path} is not an evaluation report.")
    return slot, report


def write_matrix(matrix: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "matrix.json"
    markdown_path = output_dir / "matrix.md"
    write_json(json_path, matrix)
    markdown_path.write_text(render_matrix(matrix), encoding="utf-8")
    return json_path, markdown_path


def render_matrix(matrix: dict[str, Any]) -> str:
    lines = [
        "# Evaluation matrix",
        "",
        f"OLMo 7B baseline measured: {matrix['olmo_7b_baseline_measured']}.",
        "",
        "| Slot | Role | Measured | Label | Promote |",
        "| --- | --- | --- | --- | --- |",
    ]
    for slot, row in matrix["slots"].items():
        lines.append(
            f"| {slot} | {row['role']} | {row['measured']} | {row['model_label']} | {row['promote']} |"
        )
    lines.extend(["", "Unmeasured slots have no metrics. They are not treated as a pass or a fail.", ""])
    for slot, row in matrix["slots"].items():
        if row["reason"]:
            lines.append(f"- {slot}: {row['reason']}")
    lines.append("")
    return "\n".join(lines)
