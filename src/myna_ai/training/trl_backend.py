"""Optional TRL SFT backend.

The native loop remains the tested loss-masking path. This module constructs a
TRL ``SFTTrainer`` from the same training config when a cloud job asks for
``backend: trl``. Field names differ across TRL releases, so only parameters
that exist on the installed ``SFTConfig`` are forwarded.
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
    }
    signature = inspect.signature(SFTConfig)
    accepted = {key: value for key, value in candidates.items() if key in signature.parameters}
    return SFTConfig(**accepted)


def run_trl_training(config: TrainingConfig, spec: ModelSpec) -> dict[str, Any]:
    """Run TRL SFT. Refuses CPU for any model that is not an explicit smoke run."""
    assert_training_device(spec, config)
    if spec.local_constructor == "gpt2-random":
        raise MynaAIError(
            "The TRL backend loads a Hugging Face model id. "
            "Use backend=native for the randomly initialised smoke model."
        )
    try:
        from datasets import Dataset
        from trl import SFTTrainer
    except ImportError as exc:
        raise MynaAIError("TRL training needs the train extra (trl, datasets, transformers, peft).") from exc
    from myna_ai.common.io import read_jsonl
    from myna_ai.models.chat import render_olmo3_chat
    from myna_ai.training.loop import _load_model, _load_tokenizer

    rows = []
    for record in read_jsonl(config.train_path):
        rendered = render_olmo3_chat(record["messages"], record.get("tools") or None, add_generation_prompt=False)
        rows.append({"text": rendered.text})
    dataset = Dataset.from_list(rows)
    tokenizer = _load_tokenizer(config, spec)
    model = _load_model(config, spec, tokenizer)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": build_sft_config(config),
        "train_dataset": dataset,
    }
    signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    train_output = trainer.train(resume_from_checkpoint=str(config.resume_from_checkpoint) if config.resume_from_checkpoint else None)
    trainer.save_model(str(config.output_dir / "trl_adapter"))
    return {
        "backend": "trl",
        "model_key": spec.key,
        "metrics": getattr(train_output, "metrics", {}),
        "olmo_7b_trained": spec.key == "olmo3-7b-instruct",
        "smoke_test": False,
    }
