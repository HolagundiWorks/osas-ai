"""Proposed application-neutral tool contracts.

These namespaces are examples for training and for the mock executor.
They are not confirmed MynaCAD or CoreOps APIs. ``verified`` is false on
every entry until an application integration supplies a checked schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from myna_ai.common.config import load_yaml, repo_root
from myna_ai.common.errors import ConfigError


class ToolSpec(BaseModel):
    name: str
    namespace: str
    description: str
    state_changing: bool = False
    requires_authorisation: bool = False
    verified: bool = False
    parameters: dict[str, Any]
    notes: str = ""

    def schema_document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "state_changing": self.state_changing,
            "requires_authorisation": self.requires_authorisation,
            "verified": self.verified,
        }


class ToolCatalog(BaseModel):
    namespace_status: str
    verified_against_myna_application: bool = False
    tools: list[ToolSpec]

    def by_name(self, name: str) -> ToolSpec:
        for tool in self.tools:
            if tool.name == name:
                return tool
        known = ", ".join(tool.name for tool in self.tools)
        raise ConfigError(f"Unknown tool '{name}'. Proposed tools: {known}")

    def names(self) -> set[str]:
        return {tool.name for tool in self.tools}

    def public_schemas(self) -> list[dict[str, Any]]:
        return [tool.schema_document() for tool in self.tools]


_CACHE: ToolCatalog | None = None


def catalog_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return repo_root() / "configs" / "inference" / "tools.yaml"


def load_tool_catalog(path: Path | None = None, *, cache: bool = True) -> ToolCatalog:
    global _CACHE
    if path is None and cache and _CACHE is not None:
        return _CACHE
    raw = load_yaml(catalog_path(path))
    if raw.get("verified_against_myna_application") is True:
        raise ConfigError("Tool catalog cannot claim application verification from this repository.")
    catalog = ToolCatalog.model_validate(raw)
    for tool in catalog.tools:
        if tool.verified:
            raise ConfigError(f"Tool '{tool.name}' cannot be marked verified in the example catalog.")
        if tool.state_changing and not tool.requires_authorisation:
            raise ConfigError(f"State-changing tool '{tool.name}' must require authorisation.")
        if not tool.parameters.get("type"):
            raise ConfigError(f"Tool '{tool.name}' is missing a JSON Schema type.")
    if path is None and cache:
        _CACHE = catalog
    return catalog


def tool_names(path: Path | None = None) -> set[str]:
    return load_tool_catalog(path).names()


def reset_catalog_cache() -> None:
    global _CACHE
    _CACHE = None
