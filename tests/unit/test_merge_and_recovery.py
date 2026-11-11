"""CPU checks for adapter merge and MLP recovery. These do not train OLMo 3."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.train


def test_smoke_adapter_merges_to_full_precision(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    import torch
    from safetensors.torch import load_file, save_file

    from myna_ai.common.io import write_json
    from myna_ai.export.manifest import verify_manifest
    from myna_ai.export.merge import merge_adapter_checkpoint
    from myna_ai.models.registry import require_model
    from myna_ai.training.config import TrainingConfig
    from myna_ai.training.loop import _load_model
    from myna_ai.training.tokenizer import CharTokenizer

    tokenizer = CharTokenizer()
    config = TrainingConfig(
        model_key="smoke-tiny",
        train_path=tmp_path / "train.jsonl",
        output_dir=tmp_path / "out",
        method="lora",
        max_seq_length=32,
        precision="fp32",
        allow_cpu_smoke=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        smoke_n_embd=32,
        smoke_n_layer=1,
        smoke_n_head=4,
    )
    spec = require_model("smoke-tiny")
    model = _load_model(config, spec, tokenizer)
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    tensors = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
    assert any("lora" in key for key in tensors)
    save_file(tensors, checkpoint / "model.safetensors")
    tokenizer.save(checkpoint / "char_tokenizer.json")
    write_json(checkpoint / "training_config.json", config.model_dump(mode="json"))

    manifest = merge_adapter_checkpoint(checkpoint, tmp_path / "merged")
    assert manifest["artifact_type"] == "merged_full_precision"
    assert manifest["olmo_7b_trained"] is False
    assert manifest["promotion_allowed"] is False
    assert verify_manifest(tmp_path / "merged" / "manifest.json") == []
    merged_weights = load_file(tmp_path / "merged" / "model.safetensors")
    assert merged_weights
    assert not any("lora" in key for key in merged_weights)
    from transformers import GPT2LMHeadModel

    merged = GPT2LMHeadModel.from_pretrained(tmp_path / "merged", local_files_only=True)
    merged.eval()
    logits = merged(input_ids=torch.ones(1, 4, dtype=torch.long)).logits
    assert torch.isfinite(logits).all()


def test_pruned_mlp_recovery_trains_a_copy(tmp_path: Path) -> None:
    del tmp_path
    pytest.importorskip("torch")
    import torch

    from myna_ai.compression.prune import build_tiny_olmo_mlp, prune_mlp_module
    from myna_ai.compression.recovery import recover_pruned_mlp

    torch.manual_seed(0)
    original = build_tiny_olmo_mlp()
    before = {name: value.detach().clone() for name, value in original.state_dict().items()}
    calibration = torch.randn(8, 8)
    pruned, _report = prune_mlp_module(original, percentage=0.25, calibration=calibration, alignment=1)
    recovered, recovery = recover_pruned_mlp(original, pruned, calibration, steps=40, learning_rate=0.05)
    assert recovery["launched"] is True
    assert recovery["promoted"] is False
    assert recovery["olmo_7b_trained"] is False
    assert recovery["original_module_modified"] is False
    assert recovery["loss_last"] < recovery["loss_first"]
    for name, value in original.state_dict().items():
        assert torch.equal(value, before[name])
    assert recovered.down_proj.weight.shape[1] == pruned.down_proj.weight.shape[1]
    assert recovered.down_proj.weight.shape[1] < original.down_proj.weight.shape[1]


def test_recovery_launch_resumes_smoke_training(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from myna_ai.compression.recovery import launch_recovery_finetune
    from myna_ai.datasets.prepare import prepare_dataset
    from myna_ai.training.config import TrainingConfig
    from myna_ai.training.loop import run_training

    data_dir = tmp_path / "data"
    prepare_dataset(
        adapter_name="synthetic",
        source=None,
        output_dir=data_dir,
        commercial_use=True,
        acknowledge_noncommercial_licence=False,
        seed=5,
        ratios=(1.0, 0.0, 0.0),
    )
    output = tmp_path / "run"
    config = TrainingConfig(
        model_key="smoke-tiny",
        train_path=data_dir / "train.jsonl",
        output_dir=output,
        method="lora",
        max_seq_length=128,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        learning_rate=0.02,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        precision="fp32",
        gradient_checkpointing=False,
        seed=3,
        logging_steps=1,
        save_steps=1,
        allow_cpu_smoke=True,
        backend="native",
        smoke_n_embd=32,
        smoke_n_layer=1,
        smoke_n_head=4,
    )
    first = run_training(config)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_key: smoke-tiny",
                f"train_path: {data_dir / 'train.jsonl'}",
                f"output_dir: {tmp_path / 'recovery'}",
                "method: lora",
                "max_seq_length: 128",
                "lora_r: 4",
                "lora_alpha: 8",
                "lora_dropout: 0.0",
                "learning_rate: 0.02",
                "per_device_batch_size: 1",
                "gradient_accumulation_steps: 1",
                "max_steps: 2",
                "precision: fp32",
                "gradient_checkpointing: false",
                "seed: 3",
                "logging_steps: 1",
                "save_steps: 1",
                "allow_cpu_smoke: true",
                "backend: native",
                "smoke_n_embd: 32",
                "smoke_n_layer: 1",
                "smoke_n_head: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan = launch_recovery_finetune(
        pruned_checkpoint=Path(first["checkpoint"]),
        train_config=config_path,
        output_dir=tmp_path / "recovery",
        execute=True,
        max_steps=2,
    )
    assert plan["launched"] is True
    assert plan["promoted"] is False
    assert plan["olmo_7b_trained"] is False
    assert plan["train_result"]["steps_started_from"] == 1
    assert plan["train_result"]["steps_completed"] == 2
    assert plan["train_result"]["smoke_test"] is True
    assert (tmp_path / "recovery" / "recovery_result.json").is_file()
