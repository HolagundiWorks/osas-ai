"""Run held-out cases against a generation callable and apply promotion gates."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from myna_ai.common.config import load_yaml, repo_root
from myna_ai.common.io import write_json
from myna_ai.datasets.prompts import MYNA_SYSTEM_PROMPT
from myna_ai.evaluation.cases import heldout_cases
from myna_ai.evaluation.metrics import aggregate, score_completion
from myna_ai.models.chat import render_olmo3_chat
from myna_ai.tools.catalog import load_tool_catalog

Generator = Callable[[str], str]


def default_thresholds() -> dict[str, Any]:
    return load_yaml(repo_root() / "configs" / "evaluation" / "acceptance.yaml")


def prompt_for_case(case: dict[str, Any], catalog_tools: list[dict[str, Any]]) -> str:
    names = []
    if case.get("expect_tool"):
        names.append(case["expect_tool"])
    names.extend(case.get("forbid_tools") or [])
    # Safety refusals still need the relevant schemas in view when a tool is forbidden.
    tools = [tool for tool in catalog_tools if tool["name"] in set(names)] or None
    messages = [
        {"role": "system", "content": MYNA_SYSTEM_PROMPT},
        {"role": "user", "content": case["user"]},
    ]
    return render_olmo3_chat(messages, tools, add_generation_prompt=True).text


def evaluate_generator(
    generate: Generator,
    *,
    model_label: str,
    cases: list[dict[str, Any]] | None = None,
    thresholds: dict[str, Any] | None = None,
    artifact_bytes: int | None = None,
    peak_rss_kb: int | None = None,
    peak_gpu_bytes: int | None = None,
) -> dict[str, Any]:
    catalog = load_tool_catalog()
    cases = cases if cases is not None else heldout_cases()
    thresholds = thresholds if thresholds is not None else default_thresholds()
    scored = []
    latencies = []
    for case in cases:
        prompt = prompt_for_case(case, catalog.public_schemas())
        started = time.perf_counter()
        completion = generate(prompt)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        row = score_completion(case, completion, catalog)
        row["latency_seconds"] = elapsed
        row["completion_characters"] = len(completion)
        scored.append(row)
    metrics = aggregate(scored)
    metrics["mean_latency_seconds"] = sum(latencies) / len(latencies) if latencies else None
    positive = [latency for latency in latencies if latency > 0]
    metrics["tokens_per_second"] = None
    if positive:
        characters = sum(row["completion_characters"] for row in scored)
        metrics["approximate_characters_per_second"] = characters / sum(positive)
    else:
        metrics["approximate_characters_per_second"] = None
    decision = promotion_decision(scored, metrics, thresholds)
    return {
        "model_label": model_label,
        "metrics": metrics,
        "cases": scored,
        "promotion": decision,
        "artifact_bytes": artifact_bytes,
        "peak_rss_kb": peak_rss_kb,
        "peak_gpu_bytes": peak_gpu_bytes,
        "thresholds": thresholds,
    }


def promotion_decision(
    scored: list[dict[str, Any]],
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    critical_failures = [row["id"] for row in scored if row["critical"] and not row["passed"]]
    if critical_failures:
        reasons.append("Critical engineering or revision-safety cases failed: " + ", ".join(critical_failures))
    minimums = thresholds.get("minimums") or {}
    maximums = thresholds.get("maximums") or {}
    for key, limit in minimums.items():
        value = metrics.get(key)
        if value is None:
            continue
        if value < float(limit):
            reasons.append(f"{key} {value:.3f} is below {limit}")
    for key, limit in maximums.items():
        value = metrics.get(key)
        if value is None:
            continue
        if value > float(limit):
            reasons.append(f"{key} {value:.3f} is above {limit}")
    return {"promote": not reasons, "reasons": reasons}


def reference_generator(cases: list[dict[str, Any]] | None = None) -> Generator:
    lookup = {case["user"]: case["reference_completion"] for case in (cases or heldout_cases())}

    def generate(prompt: str) -> str:
        for user, completion in lookup.items():
            if user in prompt:
                return completion
        return "I cannot tell which request this is."

    return generate


def faulty_generator(prompt: str) -> str:
    del prompt
    return "The concrete volume is 99 m3. I updated the approved estimate directly and skipped approval."


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"# Evaluation: {report['model_label']}",
        "",
        f"Promotion: {'allowed' if report['promotion']['promote'] else 'blocked'}.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Cases", "", "| Case | Critical | Passed | Tools |", "| --- | --- | --- | --- |"])
    for case in report["cases"]:
        lines.append(
            f"| {case['id']} | {case['critical']} | {case['passed']} | {', '.join(case['tool_names']) or '—'} |"
        )
    if report["promotion"]["reasons"]:
        lines.extend(["", "## Why promotion is blocked", ""])
        for reason in report["promotion"]["reasons"]:
            lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Resource notes",
            "",
            f"- Artifact bytes: {report.get('artifact_bytes')}",
            f"- Process peak RSS KB: {report.get('peak_rss_kb')}",
            f"- Peak GPU bytes: {report.get('peak_gpu_bytes')}",
            "",
            "Character rate is not a token rate. Token rate is recorded only when a tokenizer counts generated tokens.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_reports(reports: list[dict[str, Any]]) -> str:
    headers = ["metric", *[report["model_label"] for report in reports]]
    keys = sorted({key for report in reports for key in report["metrics"]})
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for key in keys:
        cells = [key]
        for report in reports:
            cells.append(str(report["metrics"].get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for report in reports:
        decision = "promote" if report["promotion"]["promote"] else "block"
        lines.append(f"- {report['model_label']}: {decision}")
    return "\n".join(lines)
