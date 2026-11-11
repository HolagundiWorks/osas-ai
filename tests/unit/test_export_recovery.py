from pathlib import Path

import pytest

from myna_ai.common.errors import ExportError, GpuRequiredError, InferenceError
from myna_ai.common.io import read_json, write_json
from myna_ai.compression.recovery import launch_recovery_finetune, plan_recovery_finetune
from myna_ai.evaluation.matrix import build_comparison_matrix, write_matrix
from myna_ai.evaluation.runner import evaluate_generator, reference_generator, write_report
from myna_ai.export.full_precision import export_full_precision
from myna_ai.export.gguf import convert_to_gguf
from myna_ai.export.manifest import verify_manifest
from myna_ai.export.merge import merge_adapter_checkpoint
from myna_ai.inference.local_runtime import LocalTransformersRuntime
from myna_ai.models.registry import require_model


def test_full_precision_export_skips_optimizer(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "optimizer.pt").write_bytes(b"optimizer")
    (source / "training_config.json").write_text("{}\n", encoding="utf-8")
    (source / "adapter").mkdir()
    (source / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    manifest = export_full_precision(
        source,
        tmp_path / "export",
        {"model_key": "smoke-tiny", "promotion_allowed": False},
    )
    names = {item["path"] for item in manifest["files"]}
    assert "model.safetensors" in names
    assert "optimizer.pt" not in names
    assert "adapter/adapter_model.safetensors" not in names
    assert manifest["optimizer_included"] is False
    assert verify_manifest(tmp_path / "export" / "manifest.json") == []


def test_matrix_leaves_unmeasured_slots_empty(tmp_path: Path) -> None:
    reference = evaluate_generator(reference_generator(), model_label="reference")
    write_report(reference, tmp_path / "reference")
    matrix = build_comparison_matrix({"B": read_json(tmp_path / "reference" / "report.json")})
    assert matrix["olmo_7b_baseline_measured"] is False
    assert matrix["slots"]["A"]["measured"] is False
    assert matrix["slots"]["A"]["metrics"] is None
    assert matrix["slots"]["B"]["measured"] is True
    assert matrix["slots"]["B"]["promote"] is True
    assert matrix["slots"]["C"]["measured"] is False
    assert matrix["slots"]["D"]["measured"] is False
    write_matrix(matrix, tmp_path / "matrix")
    text = (tmp_path / "matrix" / "matrix.md").read_text(encoding="utf-8")
    assert "OLMo 7B baseline measured: False" in text


def test_gguf_stub_is_not_a_verified_llama_cpp_load(tmp_path: Path) -> None:
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text(
        "import sys\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "outfile = Path(args[args.index('--outfile') + 1])\n"
        "outfile.write_bytes(b'not-a-llama-cpp-file')\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    report = convert_to_gguf(
        require_model("olmo3-7b-instruct"),
        model_dir=model_dir,
        output=tmp_path / "model.gguf",
        converter=converter,
        execute=True,
    )
    assert report["conversion_executed"] is True
    assert report["artifact_written"] is True
    assert report["llama_cpp_load_verified"] is False
    assert "not a verified" in report["claim"]


def test_smoke_and_unverified_qwen_are_not_converted(tmp_path: Path) -> None:
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("import sys\nraise SystemExit('should not run')\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(ExportError, match="GGUF target"):
        convert_to_gguf(
            require_model("smoke-tiny"),
            model_dir=model_dir,
            output=tmp_path / "smoke.gguf",
            converter=converter,
            execute=True,
        )
    with pytest.raises(ExportError, match="licence_verified"):
        convert_to_gguf(
            require_model("qwen2.5-7b-instruct"),
            model_dir=model_dir,
            output=tmp_path / "qwen.gguf",
            converter=converter,
            execute=True,
        )
    assert not (tmp_path / "smoke.gguf").exists()
    assert not (tmp_path / "qwen.gguf").exists()


def test_recovery_plan_does_not_launch_and_olmo_execute_refuses_cpu(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    config_path = tmp_path / "olmo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_key: olmo3-7b-instruct",
                f"train_path: {tmp_path / 'train.jsonl'}",
                f"output_dir: {tmp_path / 'recovery'}",
                "method: qlora",
                "precision: bf16",
                "allow_cpu_smoke: false",
                "max_steps: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan = plan_recovery_finetune(
        pruned_checkpoint=checkpoint,
        train_config=config_path,
        output_dir=tmp_path / "recovery",
    )
    assert plan["launched"] is False
    idle = launch_recovery_finetune(
        pruned_checkpoint=checkpoint,
        train_config=config_path,
        output_dir=tmp_path / "recovery",
        execute=False,
    )
    assert idle["launched"] is False
    assert not (tmp_path / "recovery").exists()
    with pytest.raises(GpuRequiredError):
        launch_recovery_finetune(
            pruned_checkpoint=checkpoint,
            train_config=config_path,
            output_dir=tmp_path / "recovery",
            execute=True,
            max_steps=2,
        )


def test_merged_directory_without_weights_is_not_downloaded(tmp_path: Path) -> None:
    export = tmp_path / "merged"
    export.mkdir()
    write_json(
        export / "export_metadata.json",
        {"artifact_type": "merged_full_precision", "model_key": "smoke-tiny", "smoke_test": True},
    )
    runtime = LocalTransformersRuntime(export)
    with pytest.raises(InferenceError, match="not downloaded"):
        runtime.load()


def test_trl_olmo_refuses_cpu(tmp_path: Path) -> None:
    from myna_ai.training.config import TrainingConfig
    from myna_ai.training.loop import run_training

    config = TrainingConfig(
        model_key="olmo3-7b-instruct",
        train_path=tmp_path / "train.jsonl",
        output_dir=tmp_path / "out",
        method="lora",
        precision="bf16",
        allow_cpu_smoke=False,
        backend="trl",
        max_steps=1,
    )
    with pytest.raises(GpuRequiredError):
        run_training(config)


def test_qlora_merge_is_refused(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    write_json(
        checkpoint / "training_config.json",
        {
            "model_key": "olmo3-7b-instruct",
            "train_path": str(tmp_path / "train.jsonl"),
            "output_dir": str(tmp_path / "out"),
            "method": "qlora",
            "precision": "bf16",
            "allow_cpu_smoke": False,
        },
    )
    with pytest.raises(ExportError, match="QLoRA"):
        merge_adapter_checkpoint(checkpoint, tmp_path / "merged")
