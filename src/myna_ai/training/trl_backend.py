"""Optional TRL SFT backend.

The native loop remains the path that masks every assistant span. This module
constructs a TRL ``SFTTrainer`` from the same training config. The smoke model
uses a local character tokenizer, so the run does not download a checkpoint.
TRL masks tokens before the first assistant span via ``completion_only_loss``.
Later user turns that sit inside that completion are not masked. Field names
differ across TRL releases, so only parameters that exist on the installed
``SFTConfig`` are forwarded.
"""

from __future__ import annotations

import inspect
from typing import Any

from myna_ai.common.errors import MynaAIError
from myna_ai.models.registry import ModelSpec
from myna_ai.training.config import TrainingConfig
from myna_ai.training.hardware import assert_training_device


def describe_trl() -> dict[str, Any]:
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise MynaAIError("TRL is not installed. Install the train extra: pip install -e '.[train]'") from exc
    return {
        "sft_config_parameters": sorted(inspect.signature(SFTConfig).parameters),
        "trainer": SFTTrainer.__name__,
    }


def build_sft_config(config: TrainingConfig) -> Any:
    from trl import SFTConfig

    precision_cuda = config.precision == "bf16"
    candidates = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "max_steps": config.max_steps if config.max_steps is not None else -1,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "seed": config.seed,
        "bf16": precision_cuda,
        "fp16": config.precision == "fp16",
        "gradient_checkpointing": config.gradient_checkpointing,
        "max_seq_length": config.max_seq_length,
        "max_length": config.max_seq_length,
        "packing": False,
        "report_to": "none",
        "dataset_text_field": "text",
        "completion_only_loss": True,
    }
    signature = inspect.signature(SFTConfig)
    accepted = {key: value for key, value in candidates.items() if key in signature.parameters}
    return SFTConfig(**accepted)


def run_trl_training(config: TrainingConfig, spec: ModelSpec) -> dict[str, Any]:
    """Run TRL SFT. Refuses CPU for any model that is not an explicit smoke run."""
    assert_training_device(spec, config)
    try:
        from datasets import Dataset
        from trl import SFTTrainer
    except ImportError as exc:
        raise MynaAIError("TRL training needs the train extra (trl, datasets, transformers, peft).") from exc
    from myna_ai.common.io import read_jsonl, write_json
    from myna_ai.models.chat import render_olmo3_chat
    from myna_ai.training.loop import _load_model, _load_tokenizer
    from myna_ai.training.tokenizer import CharTokenizer

    rows = _prompt_completion_rows(config.train_path, read_jsonl, render_olmo3_chat)
    if not rows:
        raise MynaAIError(f"No assistant spans to train in {config.train_path}.")
    dataset = Dataset.from_list(rows)
    tokenizer = _load_tokenizer(config, spec)
    processing = tokenizer.to_hf_tokenizer() if isinstance(tokenizer, CharTokenizer) else tokenizer
    model = _load_model(config, spec, tokenizer)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": build_sft_config(config),
        "train_dataset": dataset,
    }
    signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = processing
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = processing
    else:
        raise MynaAIError("This TRL release has no tokenizer argument on SFTTrainer.")
    trainer = SFTTrainer(**trainer_kwargs)
    resume = str(config.resume_from_checkpoint) if config.resume_from_checkpoint else None
    train_output = trainer.train(resume_from_checkpoint=resume)
    adapter_dir = config.output_dir / "trl_adapter"
    trainer.save_model(str(adapter_dir))
    metrics = getattr(train_output, "metrics", {}) or {}
    result = {
        "backend": "trl",
        "model_key": spec.key,
        "base_model": spec.hf_id,
        "metrics": metrics,
        "checkpoint": str(adapter_dir),
        "examples": len(rows),
        "loss_mask": "completion_only_from_first_assistant_span",
        "olmo_7b_trained": False,
        "smoke_test": spec.purpose == "smoke_test_only",
    }
    write_json(config.output_dir / "train_result.json", result)
    return result


def _prompt_completion_rows(path: Any, read_jsonl: Any, render_olmo3_chat: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in read_jsonl(path):
        rendered = render_olmo3_chat(record["messages"], record.get("tools") or None, add_generation_prompt=False)
        if not rendered.assistant_spans:
            continue
        start = rendered.assistant_spans[0][0]
        prompt = rendered.text[:start]
        completion = rendered.text[start:]
        if prompt and completion:
            rows.append({"prompt": prompt, "completion": completion})
    return rows
