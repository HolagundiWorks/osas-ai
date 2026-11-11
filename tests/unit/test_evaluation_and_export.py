from pathlib import Path

from myna_ai.evaluation.runner import (
    compare_reports,
    evaluate_generator,
    faulty_generator,
    reference_generator,
    write_report,
)
from myna_ai.export.gguf import assess_gguf
from myna_ai.export.manifest import export_adapter_directory, verify_manifest
from myna_ai.models.registry import require_model
from myna_ai.training.config import should_early_stop
from myna_ai.training.hardware import assert_training_device
from myna_ai.common.errors import GpuRequiredError
import pytest


def test_reference_model_can_be_promoted_and_faulty_model_cannot(tmp_path: Path) -> None:
    good = evaluate_generator(reference_generator(), model_label="reference")
    bad = evaluate_generator(faulty_generator, model_label="faulty")
    assert good["promotion"]["promote"] is True
    assert bad["promotion"]["promote"] is False
    assert any("Critical" in reason for reason in bad["promotion"]["reasons"])
    write_report(good, tmp_path / "good")
    assert (tmp_path / "good" / "report.md").is_file()
    assert "reference" in compare_reports([good, bad])


def test_olmo_gguf_is_not_claimed_as_converted() -> None:
    report = assess_gguf(require_model("olmo3-7b-instruct"))
    assert report["supported"] is True
    assert report["conversion_executed"] is False
    assert report["artifact_written"] is False
    assert report["gguf_architecture_label"] == "olmo2"
    smoke = assess_gguf(require_model("smoke-tiny"))
    assert smoke["supported"] is False


def test_manifest_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    adapter = source / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r": 4}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"not-a-real-tensor")
    manifest = export_adapter_directory(
        source,
        tmp_path / "export",
        {"artifact_type": "lora_adapter", "model_key": "smoke-tiny", "promotion_allowed": False},
    )
    assert manifest["promotion_allowed"] is False
    assert verify_manifest(tmp_path / "export" / "manifest.json") == []


def test_cpu_refusal_for_olmo_and_early_stopping() -> None:
    spec = require_model("olmo3-7b-instruct")
    from myna_ai.training.config import TrainingConfig

    config = TrainingConfig(
        model_key="olmo3-7b-instruct",
        train_path=Path("train.jsonl"),
        output_dir=Path("out"),
        method="qlora",
        precision="bf16",
        allow_cpu_smoke=False,
    )
    with pytest.raises(GpuRequiredError):
        assert_training_device(spec, config, status={"available": False, "free_gb": None, "reason": "test"})
    assert should_early_stop([1.0, 0.8, 0.8, 0.8], patience=2, min_delta=0.0) is True
    assert should_early_stop([1.0, 0.8, 0.7], patience=2, min_delta=0.0) is False
