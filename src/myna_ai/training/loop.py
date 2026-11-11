"""Native LoRA training loop.

TRL is available through ``trl_backend`` for cloud runs. This loop is the
path covered by the CPU smoke test because the assistant-token mask is
explicit and the checkpoint format is local.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from myna_ai.common.hashing import sha256_file, sha256_json
from myna_ai.common.io import read_jsonl, write_json
from myna_ai.common.logging import get_logger
from myna_ai.models.registry import ModelSpec, require_model
from myna_ai.training.collator import pad_batch, supervised_example
from myna_ai.training.config import TrainingConfig, learning_rate_at, should_early_stop
from myna_ai.training.hardware import assert_training_device
from myna_ai.training.tokenizer import CharTokenizer

LOGGER = get_logger("training")


def run_training(config: TrainingConfig) -> dict[str, Any]:
    spec = require_model(config.model_key)
    status = assert_training_device(spec, config)
    if config.backend == "trl":
        from myna_ai.training.trl_backend import run_trl_training

        return run_trl_training(config, spec)
    return _run_native(config, spec, status)


def _run_native(config: TrainingConfig, spec: ModelSpec, status: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW

    _set_seed(config.seed)
    tokenizer = _load_tokenizer(config, spec)
    examples = _load_examples(config.train_path, tokenizer, config.max_seq_length)
    if not examples:
        raise RuntimeError(f"No trainable examples in {config.train_path}")
    validation = []
    if config.validation_path and config.validation_path.is_file():
        validation = _load_examples(config.validation_path, tokenizer, config.max_seq_length)
    model = _load_model(config, spec, tokenizer)
    if config.resume_from_checkpoint:
        _load_checkpoint(model, config.resume_from_checkpoint)
    _prepare_training_mode(model, config)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
    start_step = 0
    if config.resume_from_checkpoint:
        start_step = _load_optimizer(optimizer, config.resume_from_checkpoint)
    steps_per_epoch = max(1, (len(examples) + config.per_device_batch_size - 1) // config.per_device_batch_size)
    if config.max_steps is not None:
        total_steps = config.max_steps
    else:
        total_steps = start_step + max(1, int(config.epochs * steps_per_epoch))
    if start_step >= total_steps:
        raise RuntimeError(
            f"Checkpoint step {start_step} is already at the requested max_steps {total_steps}. Nothing to resume."
        )
    device = torch.device("cuda" if status.get("available") else "cpu")
    model.to(device)
    model.train()
    if status.get("available"):
        torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    eval_losses: list[float] = []
    stopped_early = False
    step = start_step
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        batch_examples = _next_batch(examples, step, config.per_device_batch_size)
        batch = _tensor_batch(pad_batch(batch_examples, tokenizer.pad_token_id), device)
        outputs = model(**batch)
        loss = outputs.loss / config.gradient_accumulation_steps
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}. Training stopped.")
        loss.backward()
        if (step + 1) % config.gradient_accumulation_steps == 0:
            lr = learning_rate_at(step, config.learning_rate, config.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(outputs.loss.detach().cpu()))
        step += 1
        if config.logging_steps and step % config.logging_steps == 0:
            LOGGER.info("step=%s loss=%.4f rss_kb=%s", step, losses[-1], _rss_kb())
        if validation and config.eval_steps and step % config.eval_steps == 0:
            eval_loss = _evaluate(model, validation, tokenizer.pad_token_id, device)
            eval_losses.append(eval_loss)
            LOGGER.info("step=%s eval_loss=%.4f", step, eval_loss)
            if should_early_stop(eval_losses, config.early_stopping_patience, config.early_stopping_min_delta):
                stopped_early = True
                LOGGER.info("early stopping at step %s", step)
                break
        if config.save_steps and step % config.save_steps == 0:
            _save_checkpoint(model, optimizer, tokenizer, config, spec, step, losses, eval_losses)
    checkpoint = _save_checkpoint(model, optimizer, tokenizer, config, spec, step, losses, eval_losses)
    result = {
        "model_key": spec.key,
        "base_model": spec.hf_id,
        "method": config.method,
        "backend": "native",
        "device": str(device),
        "cuda": status,
        "steps_completed": step,
        "steps_started_from": start_step,
        "losses": losses,
        "eval_losses": eval_losses,
        "early_stopped": stopped_early,
        "checkpoint": str(checkpoint),
        "peak_rss_kb": _rss_kb(),
        "peak_gpu_bytes": _peak_gpu_bytes(),
        "git_commit": _git_commit(),
        "config_sha256": sha256_json(config.model_dump(mode="json")),
        "smoke_test": spec.purpose == "smoke_test_only",
        "olmo_7b_trained": False,
    }
    write_json(config.output_dir / "train_result.json", result)
    return result


def _set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        from accelerate.utils import set_seed as accelerate_seed

        accelerate_seed(seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenizer(config: TrainingConfig, spec: ModelSpec) -> Any:
    if spec.local_constructor == "gpt2-random":
        if config.resume_from_checkpoint:
            path = config.resume_from_checkpoint / "char_tokenizer.json"
            if path.is_file():
                return CharTokenizer.load(path)
        return CharTokenizer()
    from transformers import AutoTokenizer

    source = str(config.model_path or spec.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, local_files_only=config.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_examples(path: Path, tokenizer: Any, max_length: int) -> list[dict[str, list[int]]]:
    examples = []
    skipped = 0
    for record in read_jsonl(path):
        tools = record.get("tools") or None
        example = supervised_example(tokenizer, record["messages"], tools, max_length)
        if example is None:
            skipped += 1
            continue
        examples.append(example)
    if skipped:
        LOGGER.info("skipped %s examples with no assistant tokens inside the sequence limit", skipped)
    return examples


def _load_model(config: TrainingConfig, spec: ModelSpec, tokenizer: Any) -> Any:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    if spec.local_constructor == "gpt2-random":
        from transformers import GPT2Config, GPT2LMHeadModel

        model_config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=config.max_seq_length,
            n_embd=config.smoke_n_embd,
            n_layer=config.smoke_n_layer,
            n_head=config.smoke_n_head,
            n_inner=config.smoke_n_embd * 4,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = GPT2LMHeadModel(model_config)
    else:
        from transformers import AutoModelForCausalLM

        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[config.precision]
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": config.local_files_only,
            "trust_remote_code": False,
        }
        if config.method == "qlora":
            from transformers import BitsAndBytesConfig

            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise GpuRequiredErrorBits() from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = "auto"
        source = str(config.model_path or spec.hf_id)
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        if config.method == "qlora":
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
    if getattr(model.config, "use_cache", None):
        model.config.use_cache = False
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules or spec.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        fan_in_fan_out=spec.fan_in_fan_out,
    )
    return get_peft_model(model, lora)


def GpuRequiredErrorBits() -> Exception:
    from myna_ai.common.errors import GpuRequiredError

    return GpuRequiredError(
        "QLoRA requires bitsandbytes and a CUDA GPU. Install the quant extra on the training machine. "
        "Training was not started, and the run did not fall back to full-precision CPU training."
    )


def _prepare_training_mode(model: Any, config: TrainingConfig) -> None:
    if not config.gradient_checkpointing:
        return
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable()
    except Exception as exc:
        raise RuntimeError(f"Gradient checkpointing was requested and failed: {exc}") from exc


def _tensor_batch(batch: dict[str, list[list[int]]], device: Any) -> dict[str, Any]:
    import torch

    return {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in batch.items()}


def _next_batch(examples: list[dict[str, list[int]]], step: int, size: int) -> list[dict[str, list[int]]]:
    start = (step * size) % len(examples)
    selected = []
    for offset in range(size):
        selected.append(examples[(start + offset) % len(examples)])
    return selected


def _evaluate(model: Any, examples: list[dict[str, list[int]]], pad_id: int, device: Any) -> float:
    import torch

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        batch = _tensor_batch(pad_batch(examples, pad_id), device)
        outputs = model(**batch)
        total += float(outputs.loss) * batch["input_ids"].size(0)
        count += batch["input_ids"].size(0)
    model.train()
    return total / max(count, 1)


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    tokenizer: Any,
    config: TrainingConfig,
    spec: ModelSpec,
    step: int,
    losses: list[float],
    eval_losses: list[float],
) -> Path:
    import torch
    from safetensors.torch import save_file

    checkpoint = config.output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint / "adapter"
    model.save_pretrained(adapter_dir)
    tensors = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
    save_file(tensors, str(checkpoint / "model.safetensors"))
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    if isinstance(tokenizer, CharTokenizer):
        tokenizer.save(checkpoint / "char_tokenizer.json")
    write_json(
        checkpoint / "trainer_state.json",
        {
            "global_step": step,
            "model_key": spec.key,
            "losses": losses,
            "eval_losses": eval_losses,
            "smoke_test": spec.purpose == "smoke_test_only",
            "olmo_7b_trained": False,
        },
    )
    write_json(checkpoint / "training_config.json", json.loads(config.model_dump_json()))
    files = {
        "model.safetensors": sha256_file(checkpoint / "model.safetensors"),
        "optimizer.pt": sha256_file(checkpoint / "optimizer.pt"),
    }
    write_json(checkpoint / "checksums.json", files)
    return checkpoint


def _load_checkpoint(model: Any, checkpoint: Path) -> None:
    from safetensors.torch import load_file

    state = load_file(str(checkpoint / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected tensors: {unexpected[:8]}")
    if missing:
        LOGGER.info("checkpoint left %s tensors at their initial values", len(missing))


def _load_optimizer(optimizer: Any, checkpoint: Path) -> int:
    import torch

    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True))
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["global_step"])


def _rss_kb() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _peak_gpu_bytes() -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
