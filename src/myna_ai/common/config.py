"""YAML loading. Environment values are read only through explicit placeholders."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from myna_ai.common.errors import ConfigError

_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


def repo_root() -> Path:
    """Return the repository root that contains pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ConfigError("Could not locate the repository root from the installed package.")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file must be a mapping: {path}")
    return _expand(data)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value
