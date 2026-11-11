"""Dataset adapter interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class DatasetAdapter(Protocol):
    name: str
    licence: str

    def inspect(self, source: Path) -> dict[str, Any]:
        """Return a JSON-serialisable description of a local source."""

    def load_records(self, source: Path, options: dict[str, Any]) -> list[dict[str, Any]]:
        """Return canonical conversation dictionaries. They are validated later."""
