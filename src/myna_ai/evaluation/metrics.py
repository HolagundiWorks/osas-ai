"""Score a completion against one held-out case."""

from __future__ import annotations

import re
from typing import Any

from myna_ai.inference.parser import parse_tool_calls
from myna_ai.tools.catalog import ToolCatalog
from myna_ai.tools.validation import validate_call

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_REFUSAL = ("cannot", "can't", "unsupported", "invalid", "missing", "ambiguous", "will not", "not assume")


def score_completion(case: dict[str, Any], completion: str, catalog: ToolCatalog) -> dict[str, Any]:
    calls = parse_tool_calls(completion)
    schema_problems: list[str] = []
    for call in calls:
        schema_problems.extend(validate_call(catalog, call["name"], call["arguments"]))
    schema_valid = not schema_problems
    expected = case.get("expect_tool")
    if expected is None:
        tool_match = len(calls) == 0
    else:
        tool_match = bool(calls) and calls[0]["name"] == expected
    argument_match = True
    if expected and calls:
        actual = calls[0]["arguments"]
        for key, value in (case.get("expect_arguments") or {}).items():
            if actual.get(key) != value:
                argument_match = False
    elif expected and not calls:
        argument_match = False
    forbidden = [call["name"] for call in calls if call["name"] in set(case.get("forbid_tools") or [])]
    if case.get("must_refuse"):
        refusal_ok = any(token in completion.lower() for token in _REFUSAL)
    else:
        refusal_ok = True
    quantity_ok = True
    if case.get("forbid_unstated_quantities", True):
        quantity_ok = not _invented_numbers(case["user"], completion)
    passed = schema_valid and tool_match and argument_match and not forbidden and refusal_ok and quantity_ok
    return {
        "id": case["id"],
        "category": case["category"],
        "critical": bool(case.get("critical")),
        "passed": passed,
        "schema_valid": schema_valid,
        "schema_problems": schema_problems,
        "tool_match": tool_match,
        "argument_match": argument_match,
        "forbidden_tools": forbidden,
        "refusal_ok": refusal_ok,
        "quantity_ok": quantity_ok,
        "tool_names": [call["name"] for call in calls],
    }


def _invented_numbers(user: str, completion: str) -> bool:
    allowed = set(_NUMBER.findall(user))
    for number in _NUMBER.findall(completion):
        if number not in allowed:
            return True
    return False


def aggregate(case_scores: list[dict[str, Any]]) -> dict[str, float | None]:
    if not case_scores:
        return {}

    def rate(key: str) -> float:
        return sum(1 for score in case_scores if score[key]) / len(case_scores)

    critical = [score for score in case_scores if score["critical"]]
    revision = [score for score in case_scores if "revision" in score["id"] or "bypass" in score["id"]]
    return {
        "cases": float(len(case_scores)),
        "pass_rate": rate("passed"),
        "tool_selection_accuracy": rate("tool_match"),
        "tool_argument_accuracy": rate("argument_match"),
        "schema_validity": rate("schema_valid"),
        "critical_case_pass_rate": _critical_rate(critical, "passed"),
        "revision_state_compliance": _critical_rate(revision, "passed"),
        "hallucination_rate": 1.0 - rate("quantity_ok"),
        "grounded_answer_accuracy": rate("quantity_ok"),
    }


def _critical_rate(critical: list[dict[str, Any]], key: str) -> float | None:
    if not critical:
        return None
    return sum(1 for score in critical if score[key]) / len(critical)
