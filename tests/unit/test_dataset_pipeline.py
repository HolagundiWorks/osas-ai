import json
from pathlib import Path

import pytest

from myna_ai.common.errors import DatasetLicenceError
from myna_ai.datasets.cad_environments import INDEX_COLUMNS, CadEnvironmentsAdapter
from myna_ai.datasets.dedup import drop_exact_duplicates
from myna_ai.datasets.events import summarise_events
from myna_ai.datasets.prepare import prepare_dataset
from myna_ai.datasets.split import apply_splits, assign_split
from myna_ai.evaluation.cases import heldout_cases

FIXTURE = Path("tests/fixtures/cad")


def test_index_columns_match_verified_parquet_schema() -> None:
    rows = json.loads((FIXTURE / "workflows.json").read_text(encoding="utf-8"))
    assert set(INDEX_COLUMNS) <= set(rows[0])


def test_events_are_not_executable_and_drop_clipboard() -> None:
    summary = summarise_events(FIXTURE / "autocad" / "wf-line" / "events.jsonl")
    assert summary["executable"] is False
    assert summary["counts_by_type"]["click"] == 1
    assert summary["counts_by_type"]["clipboard"] == 1
    serialised = json.dumps(summary)
    assert "do-not-train-on-this" not in serialised
    assert "12" not in json.dumps(summary["counts_by_type"])


def test_cad_adapter_filters_software_and_withholds_clicks() -> None:
    records = CadEnvironmentsAdapter().load_records(FIXTURE / "workflows.json", {"asset_root": FIXTURE})
    assert [record["id"] for record in records] == ["cadenv-wf-line"]
    text = json.dumps([record["messages"] for record in records])
    assert "do-not-train-on-this" not in text
    assert "Clicked at coordinates" not in text
    assistant = records[0]["messages"][-1]["content"]
    assert "myna_api" in assistant
    assert "null" in assistant
    assert records[0]["api_status"] == "example_schema"
    assert records[0]["source"]["licence"] == "cc-by-nc-4.0"


def test_commercial_use_of_cad_environments_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetLicenceError):
        prepare_dataset(
            adapter_name="cad_environments",
            source=FIXTURE / "workflows.json",
            output_dir=tmp_path / "blocked",
            commercial_use=True,
            acknowledge_noncommercial_licence=False,
            adapter_options={"asset_root": str(FIXTURE)},
        )


def test_research_prepare_writes_splits_without_workflow_leakage(tmp_path: Path) -> None:
    report = prepare_dataset(
        adapter_name="cad_environments",
        source=FIXTURE / "workflows.json",
        output_dir=tmp_path / "research",
        commercial_use=False,
        acknowledge_noncommercial_licence=True,
        adapter_options={"asset_root": str(FIXTURE)},
        seed=3,
    )
    assert report["rejected_count"] == 0
    assert report["input_record_count"] == 1
    written = list((tmp_path / "research").glob("*.jsonl"))
    assert written
    combined = []
    for path in written:
        combined.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    assert {row["source"]["workflow_id"] for row in combined} == {"wf-line"}


def test_synthetic_prepare_does_not_contain_heldout_prompts(tmp_path: Path) -> None:
    prepare_dataset(
        adapter_name="synthetic",
        source=None,
        output_dir=tmp_path / "synthetic",
        commercial_use=True,
        acknowledge_noncommercial_licence=False,
        seed=1,
        ratios=(0.7, 0.15, 0.15),
    )
    users = []
    for name in ("train", "validation", "test"):
        path = tmp_path / "synthetic" / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            users.extend(message["content"] for message in record["messages"] if message["role"] == "user")
    for case in heldout_cases():
        assert case["user"] not in users


def test_duplicates_and_split_stick_to_workflow() -> None:
    record = {"id": "a", "category": "cad", "messages": [{"role": "user", "content": "same"}], "source": {"workflow_id": "w"}}
    copy = {"id": "b", "category": "cad", "messages": [{"role": "user", "content": "same"}], "source": {"workflow_id": "w"}}
    kept, dropped = drop_exact_duplicates([record, copy])
    assert [item["id"] for item in kept] == ["a"]
    assert dropped[0]["duplicate_of"] == "a"
    first = {"id": "1", "source": {"workflow_id": "shared"}, "messages": []}
    second = {"id": "2", "source": {"workflow_id": "shared"}, "messages": []}
    splits = apply_splits([first, second], seed=9)
    homes = [row["split"] for group in splits.values() for row in group]
    assert homes[0] == homes[1]
    assert assign_split("shared", seed=9) == homes[0]


def test_parametric_adapter(tmp_path: Path) -> None:
    report = prepare_dataset(
        adapter_name="parametric_instructions",
        source=Path("tests/fixtures/parametric.jsonl"),
        output_dir=tmp_path / "param",
        commercial_use=True,
        acknowledge_noncommercial_licence=False,
    )
    assert report["rejected_count"] == 0
    assert report["input_record_count"] == 1
