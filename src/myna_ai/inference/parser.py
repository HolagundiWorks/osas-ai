"""Parse OLMo function-call text and JSON tool calls."""

from __future__ import annotations

import json
import re
from typing import Any

_BLOCK = re.compile(r"<function_calls>(.*?)</function_calls>", re.DOTALL)
_CALL = re.compile(r"^([A-Za-z_][A-Za-z0-9_\.]*)\((.*)\)\s*$", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in _BLOCK.finditer(text):
        for line in match.group(1).splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parsed = _parse_call(stripped)
            if parsed is not None:
                parsed["id"] = f"call-{len(calls) + 1}"
                calls.append(parsed)
    if calls:
        return calls
    for match in _FENCE.finditer(text):
        calls.extend(_parse_json_blob(match.group(1)))
    if not calls:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            calls.extend(_parse_json_blob(stripped))
    for index, call in enumerate(calls, start=1):
        call.setdefault("id", f"call-{index}")
    return calls


def _parse_call(line: str) -> dict[str, Any] | None:
    match = _CALL.match(line)
    if match is None:
        return None
    name, argument_text = match.group(1), match.group(2).strip()
    return {"name": name, "arguments": {} if not argument_text else _parse_arguments(argument_text)}


def _parse_arguments(argument_text: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for key, raw in _split_arguments(argument_text):
        arguments[key] = json.loads(raw)
    return arguments


def _split_arguments(argument_text: str) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    depth = 0
    in_string = False
    escape = False
    start = 0
    segments: list[str] = []
    for index, character in enumerate(argument_text):
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
        elif character == "," and depth == 0:
            segments.append(argument_text[start:index])
            start = index + 1
    segments.append(argument_text[start:])
    for segment in segments:
        if "=" not in segment:
            raise ValueError(f"Malformed tool argument: {segment}")
        key, raw = segment.split("=", 1)
        pieces.append((key.strip(), raw.strip()))
    return pieces


def _parse_json_blob(blob: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return []
    raw_calls = payload.get("tool_calls") if isinstance(payload, dict) else None
    if not isinstance(raw_calls, list):
        return []
    calls = []
    for item in raw_calls:
        if not isinstance(item, dict) or "name" not in item:
            continue
        arguments = item.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append({"name": item["name"], "arguments": arguments})
    return calls
