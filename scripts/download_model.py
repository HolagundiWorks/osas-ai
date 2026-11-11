"""Download a registered Hugging Face model into a local directory.

The default is a dry run. A real download requires --execute and uses HF_TOKEN
from the environment when it is set. The token is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from myna_ai.common.logging import configure_logging, redact
from myna_ai.models.registry import require_model


def plan(model_key: str, destination: Path) -> dict:
    spec = require_model(model_key)
    if spec.hf_id is None:
        return {
            "model_key": spec.key,
            "download": False,
            "reason": "This entry is constructed locally and has no Hugging Face weights.",
        }
    return {
        "model_key": spec.key,
        "repo_id": spec.hf_id,
        "destination": str(destination),
        "licence": spec.licence,
        "licence_verified": spec.licence_verified,
        "executed": False,
        "local_files_only_after_download": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or download a registered model.")
    parser.add_argument("--model-key", default="olmo3-7b-instruct")
    parser.add_argument("--destination", type=Path, default=Path("models") / "olmo3-7b-instruct")
    parser.add_argument("--execute", action="store_true", help="Download weights. Omit this for a plan.")
    args = parser.parse_args()
    configure_logging()
    report = plan(args.model_key, args.destination)
    if args.execute and report.get("repo_id"):
        from huggingface_hub import snapshot_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        args.destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=report["repo_id"], local_dir=args.destination, token=token)
        report["executed"] = True
    print(redact(json.dumps(report, indent=2)))


if __name__ == "__main__":
    main()
