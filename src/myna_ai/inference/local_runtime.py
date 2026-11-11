"""Load a local checkpoint for offline generation.

The smoke checkpoint is a tiny random GPT-2 plus a LoRA state dict. OLMo 3 is
loaded only from a local directory when ``local_files_only`` is true, so a
missing download fails instead of fetching weights.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from myna_ai import __version__
from myna_ai.common.errors import CancelledError, InferenceError, InferenceTimeoutError
from myna_ai.common.io import read_json


class LocalTransformersRuntime:
    def __init__(self, checkpoint: Path, training_config: Path | None = None) -> None:
        self.checkpoint = checkpoint
        self.training_config = training_config
        self._model = None
        self._tokenizer = None
        self._version: dict[str, Any] = {
            "model_key": "unknown",
            "offline": True,
            "checkpoint": str(checkpoint),
            "myna_ai_version": __version__,
        }

    def load(self) -> None:
        metadata_path = self.checkpoint / "export_metadata.json"
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            if metadata.get("artifact_type") == "merged_full_precision":
                self._load_merged_directory(metadata)
                return
        self._load_training_checkpoint()

    def _load_training_checkpoint(self) -> None:
        import torch

        from myna_ai.training.config import TrainingConfig
        from myna_ai.training.loop import _load_model
        from myna_ai.training.tokenizer import CharTokenizer

        state_path = self.checkpoint / "trainer_state.json"
        if not state_path.is_file():
            raise InferenceError(
                f"No trainer_state.json in {self.checkpoint}. "
                "A merged export needs export_metadata.json with artifact_type merged_full_precision. "
                "Weights are not downloaded."
            )
        state = read_json(state_path)
        config_payload = read_json(self.checkpoint / "training_config.json")
        config = TrainingConfig.model_validate(config_payload)
        self._max_positions = config.max_seq_length
        if (self.checkpoint / "char_tokenizer.json").is_file():
            tokenizer = CharTokenizer.load(self.checkpoint / "char_tokenizer.json")
        else:
            raise InferenceError(
                "This runtime currently reloads the character-tokenizer smoke checkpoint. "
                "An OLMo tokenizer must be loaded with local_files_only from a downloaded model directory."
            )
        from myna_ai.models.registry import require_model

        spec = require_model(config.model_key)
        model = _load_model(config, spec, tokenizer)
        from safetensors.torch import load_file

        missing, unexpected = model.load_state_dict(load_file(str(self.checkpoint / "model.safetensors")), strict=False)
        if unexpected:
            raise InferenceError(f"Unexpected tensors in checkpoint: {unexpected[:5]}")
        if missing:
            raise InferenceError(f"Checkpoint is missing {len(missing)} tensors, including {missing[:3]}")
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device("cpu")
        model.to(self._device)
        self._version = {
            "model_key": config.model_key,
            "base_model_id": spec.hf_id,
            "checkpoint": str(self.checkpoint),
            "global_step": state.get("global_step"),
            "offline": True,
            "smoke_test": state.get("smoke_test", False),
            "olmo_7b_trained": False,
            "myna_ai_version": __version__,
        }

    def _load_merged_directory(self, metadata: dict[str, Any]) -> None:
        import torch

        config_file = self.checkpoint / "config.json"
        weights = self.checkpoint / "model.safetensors"
        if not config_file.is_file() or not weights.is_file():
            raise InferenceError(
                f"Merged export {self.checkpoint} is missing config.json or model.safetensors. "
                "Weights are not downloaded."
            )
        tokenizer = self._load_offline_tokenizer()
        with _hub_offline():
            try:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception as exc:
                raise InferenceError(
                    f"Could not load {self.checkpoint} from local files. No weights were downloaded."
                ) from exc
        if getattr(model.config, "use_cache", None) is False:
            model.config.use_cache = True
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device("cpu")
        model.to(self._device)
        positions = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 2048)
        self._max_positions = int(positions)
        self._version = {
            "model_key": metadata.get("model_key", "unknown"),
            "base_model_id": metadata.get("base_model"),
            "artifact_type": "merged_full_precision",
            "checkpoint": str(self.checkpoint),
            "offline": True,
            "smoke_test": bool(metadata.get("smoke_test")),
            "olmo_7b_trained": False,
            "myna_ai_version": __version__,
        }

    def _load_offline_tokenizer(self) -> Any:
        from myna_ai.training.tokenizer import CharTokenizer

        char_path = self.checkpoint / "char_tokenizer.json"
        if char_path.is_file():
            return CharTokenizer.load(char_path)
        if (self.checkpoint / "tokenizer_config.json").is_file():
            with _hub_offline():
                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
                except Exception as exc:
                    raise InferenceError(
                        f"Could not load the tokenizer in {self.checkpoint}. No tokenizer was downloaded."
                    ) from exc
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        raise InferenceError(
            f"No char_tokenizer.json or tokenizer_config.json in {self.checkpoint}. "
            "Refusing to download a tokenizer."
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        cancel: threading.Event | None,
        timeout_s: float | None,
    ) -> str:
        if self._model is None or self._tokenizer is None:
            self.load()
        import time

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        if cancel is not None and cancel.is_set():
            raise CancelledError("Inference was cancelled.")
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s
        tokenizer = self._tokenizer
        encoded = tokenizer(prompt, return_offsets_mapping=False, add_special_tokens=False)
        room = max(1, self._max_positions - max_new_tokens)
        token_ids = encoded["input_ids"][-room:]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)

        class _Stop(StoppingCriteria):
            def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> bool:
                del input_ids, scores, kwargs
                if cancel is not None and cancel.is_set():
                    return True
                if deadline is not None and time.perf_counter() > deadline:
                    return True
                return False

        assert self._model is not None
        output = self._model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([_Stop()]),
        )
        if cancel is not None and cancel.is_set():
            raise CancelledError("Inference was cancelled.")
        if deadline is not None and time.perf_counter() > deadline:
            raise InferenceTimeoutError("Inference timed out.")
        new_tokens = output[0, input_ids.shape[1] :].tolist()
        return tokenizer.decode(new_tokens, skip_special_tokens=False)

    def model_version(self) -> dict[str, Any]:
        if self._model is None:
            self.load()
        return dict(self._version)


@contextmanager
def _hub_offline() -> Iterator[None]:
    """Keep a load from fetching missing files."""
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
