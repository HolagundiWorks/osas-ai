"""Versioned export manifests and checksum verification."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myna_ai import __version__
from myna_ai.common.errors import ExportError
from myna_ai.common.hashing import sha256_file, sha256_json
from myna_ai.common.io import write_json


def export_adapter_directory(source: Path, destination: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    adapter = source / "adapter" if (source / "adapter").is_dir() else source
    config = adapter / "adapter_config.json"
    if not config.is_file():
        raise ExportError(
            f"No PEFT adapter_config.json under {source}. "
            "Refusing to write an export manifest for a directory that is not an adapter."
        )
    if destination.exists():
        raise ExportError(f"Export destination already exists: {destination}")
    destination.mkdir(parents=True)
    for path in adapter.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    manifest = build_manifest(destination, metadata)
    write_json(destination / "manifest.json", manifest)
    return manifest


def build_manifest(artifact_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(artifact_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "format_version": "1",
        "myna_ai_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "metadata_sha256": sha256_json(metadata),
        **metadata,
    }
    return manifest


def verify_manifest(manifest_path: Path) -> list[str]:
    from myna_ai.common.io import read_json

    manifest = read_json(manifest_path)
    problems = []
    root = manifest_path.parent
    for item in manifest.get("files") or []:
        path = root / item["path"]
        if not path.is_file():
            problems.append(f"missing {item['path']}")
            continue
        digest = sha256_file(path)
        if digest != item["sha256"]:
            problems.append(f"checksum mismatch {item['path']}")
        if path.stat().st_size != item["bytes"]:
            problems.append(f"size mismatch {item['path']}")
    return problems
