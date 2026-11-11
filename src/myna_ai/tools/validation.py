"""Validate tool-call arguments against the proposed JSON schemas."""

from __future__ import annotations

from typing import Any

import jsonschema

from myna_ai.tools.catalog import ToolCatalog, ToolSpec


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    try:
        jsonschema.validate(instance=arguments, schema=spec.parameters)
    except jsonschema.ValidationError as exc:
        return [exc.message]
    return []


def validate_call(catalog: ToolCatalog, name: str, arguments: dict[str, Any]) -> list[str]:
    try:
        spec = catalog.by_name(name)
    except Exception as exc:  # ConfigError
        return [str(exc)]
    problems = validate_arguments(spec, arguments)
    if spec.verified:
        problems.append("Verified tools are not registered.")
    return problems
