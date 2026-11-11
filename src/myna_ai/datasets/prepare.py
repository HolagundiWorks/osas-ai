"""Build validated JSONL splits from a registered adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from myna_ai.common.hashing import sha256_json
from myna_ai.common.io import write_json, write_jsonl
from myna_ai.datasets.adapters import DatasetAdapter
from myna_ai.datasets.cad_environments import CadEnvironmentsAdapter
from myna_ai.datasets.dedup import drop_exact_duplicates
from myna_ai.datasets.licence import assert_licence_allows_use
from myna_ai.datasets.parametric import ParametricInstructionAdapter
from myna_ai.datasets.split import apply_splits
from myna_ai.datasets.synthetic import build_synthetic_corpus
from myna_ai.datasets.validate import validate_records

ADAPTERS: dict[str, DatasetAdapter] = {
    "cad_environments": CadEnvironmentsAdapter(),
    "parametric_instructions": ParametricInstructionAdapter(),
}


def prepare_dataset(
    *,
    adapter_name: str,
    source: Path | None,
    output_dir: Path,
    commercial_use: bool,
    acknowledge_noncommercial_licence: bool,
    seed: int = 17,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    adapter_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(adapter_options or {})
    if adapter_name == "synthetic":
        records = build_synthetic_corpus()
        licence = "apache-2.0"
        dataset_id = "myna-ai-synthetic"
        inspection: dict[str, Any] = {"adapter": "synthetic", "rows": len(records), "licence": licence}
    else:
        if source is None:
            raise FileNotFoundError("This adapter needs a local source path.")
        adapter = ADAPTERS[adapter_name]
        licence = adapter.licence
        dataset_id = adapter_name
        inspection = adapter.inspect(source)
        if isinstance(inspection.get("licence"), str):
            licence = str(inspection["licence"])
        if adapter_name == "cad_environments":
            dataset_id = "markov-ai/cad-environments"
            licence = "cc-by-nc-4.0"
        records = adapter.load_records(source, options)
    assert_licence_allows_use(
        licence,
        commercial_use=commercial_use,
        acknowledge_noncommercial_licence=acknowledge_noncommercial_licence,
        dataset_id=dataset_id,
    )
    kept, duplicates = drop_exact_duplicates(records)
    accepted, rejected = validate_records(kept)
    payloads = [record.to_json_dict() for record in accepted]
    splits = apply_splits(payloads, seed=seed, ratios=ratios)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    provenance = {
        "dataset_id": dataset_id,
        "licence": licence,
        "adapter": adapter_name,
        "commercial_use": commercial_use,
        "seed": seed,
        "ratios": list(ratios),
        "input_record_count": len(records),
        "duplicate_count": len(duplicates),
        "rejected_count": len(rejected),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "content_sha256": sha256_json(payloads),
        "inspection": inspection,
    }
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "duplicates.json", duplicates)
    write_json(output_dir / "rejected.json", rejected)
    return provenance
