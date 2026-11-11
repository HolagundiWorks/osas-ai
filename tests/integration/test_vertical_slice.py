"""CPU vertical slice. A pass does not mean OLMo 3 7B was trained."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.train


def test_smoke_train_eval_export_and_mock_tool(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    from myna_ai.compression.prune import assess_olmo3_pruning, build_tiny_olmo_mlp, prune_mlp_module
    from myna_ai.compression.quantize import dynamic_int8_size_report
    from myna_ai.datasets.prepare import prepare_dataset
    from myna_ai.export.manifest import export_adapter_directory, verify_manifest
    from myna_ai.inference.client import InferenceRequest, LocalInferenceClient, ScriptedRuntime
    from myna_ai.inference.local_runtime import LocalTransformersRuntime
    from myna_ai.tools.mock_executor import Authorisation, ProjectState
    from myna_ai.training.config import TrainingConfig
    from myna_ai.training.loop import run_training

    data_dir = tmp_path / "data"
    prepare_dataset(
        adapter_name="synthetic",
        source=None,
        output_dir=data_dir,
        commercial_use=True,
        acknowledge_noncommercial_licence=False,
        seed=4,
        ratios=(1.0, 0.0, 0.0),
    )
    output = tmp_path / "run"
    config = TrainingConfig(
        model_key="smoke-tiny",
        train_path=data_dir / "train.jsonl",
        validation_path=None,
        output_dir=output,
        method="lora",
        max_seq_length=256,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        learning_rate=0.02,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=6,
        precision="fp32",
        gradient_checkpointing=False,
        seed=7,
        logging_steps=1,
        save_steps=3,
        allow_cpu_smoke=True,
        backend="native",
    )
    first = run_training(config)
    assert first["smoke_test"] is True
    assert first["olmo_7b_trained"] is False
    assert first["steps_completed"] == 6
    assert all(loss == loss for loss in first["losses"])  # finite
    assert first["losses"][-1] < first["losses"][0]

    resumed = run_training(
        config.model_copy(update={"max_steps": 8, "resume_from_checkpoint": Path(first["checkpoint"])})
    )
    assert resumed["steps_started_from"] == 6
    assert resumed["steps_completed"] == 8

    manifest = export_adapter_directory(
        Path(resumed["checkpoint"]),
        tmp_path / "export",
        {
            "artifact_type": "lora_adapter",
            "model_key": "smoke-tiny",
            "promotion_allowed": False,
            "notes": "Smoke adapter. Not an OLMo 7B artifact.",
        },
    )
    assert verify_manifest(tmp_path / "export" / "manifest.json") == []
    assert manifest["model_key"] == "smoke-tiny"

    runtime = LocalTransformersRuntime(Path(resumed["checkpoint"]))
    text = runtime.generate("Grid line. ", max_new_tokens=8, cancel=None, timeout_s=30)
    assert isinstance(text, str)
    version = runtime.model_version()
    assert version["offline"] is True
    assert version["olmo_7b_trained"] is False

    project = ProjectState(project_id="tower-a")
    completion = (
        '<function_calls>cad.create_geometry(project_id="tower-a", kind="line", layer="A-GRID", '
        'units="mm", geometry={"start": [0, 0], "end": [4500, 0]})</function_calls>'
    )
    result = LocalInferenceClient(ScriptedRuntime(completion, model_key="smoke-tiny")).complete(
        InferenceRequest(
            messages=[{"role": "user", "content": "Create the grid line."}],
            execute_tools=True,
            project=project,
            authorisation=Authorisation(allow=True, actor="tester"),
        )
    )
    assert result.validation_problems == []
    assert result.executions[0].output["persisted"] is False
    assert project.entities[0]["layer"] == "A-GRID"
    assert result.model_version["offline"] is True

    import torch

    original = build_tiny_olmo_mlp()
    before = original.gate_proj.weight.detach().clone()
    calibration = torch.randn(4, 8)
    _pruned, report = prune_mlp_module(original, percentage=0.10, calibration=calibration, alignment=1)
    assert torch.equal(original.gate_proj.weight, before)
    assert report["promoted"] is False
    assert report["latency_claim"] is None
    compatibility = assess_olmo3_pruning()
    assert compatibility["gguf_pruned_export"] == "unsupported_until_conversion_test"
    quant = dynamic_int8_size_report(build_tiny_olmo_mlp())
    assert quant["qlora"] is False
    assert quant["gguf"] is False
    assert quant["original_bytes"] > 0


def test_transformers_registers_olmo3_and_trl_config_builds() -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("trl")
    from transformers.models.olmo3.configuration_olmo3 import Olmo3Config

    from myna_ai.training.config import TrainingConfig
    from myna_ai.training.trl_backend import build_sft_config

    assert Olmo3Config().model_type == "olmo3"
    config = TrainingConfig(
        model_key="smoke-tiny",
        train_path=Path("train.jsonl"),
        output_dir=Path("out"),
        method="lora",
        precision="fp32",
        max_steps=1,
        max_seq_length=128,
        allow_cpu_smoke=True,
    )
    sft_config = build_sft_config(config)
    assert sft_config.max_steps == 1
    assert sft_config.bf16 is False
    assert sft_config.max_length == 128
    assert sft_config.completion_only_loss is True
