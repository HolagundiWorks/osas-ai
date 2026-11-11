"""Tokenizers used by the training loop.

The smoke tokenizer is character-level and prefix-stable, so assistant-span
masking is exact. Hugging Face tokenizers are accepted when they return
offset mappings (``use_fast=True``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CharTokenizer:
    """Fixed alphabet tokenizer. No network and no pretrained files."""

    def __init__(self) -> None:
        characters = "".join(chr(code) for code in range(32, 127)) + "\n\t"
        self.pad_token = "<|pad|>"
        self.eos_token = "<|endoftext|>"
        self.unk_token = "<|unk|>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self._itos = ["<|pad|>", "<|endoftext|>", "<|unk|>", *characters]
        self._stoi = {token: index for index, token in enumerate(self._itos)}
        self.vocab_size = 128
        if len(self._itos) > self.vocab_size:
            raise RuntimeError("Character vocabulary exceeds the reserved model vocab.")

    def __call__(self, text: str, return_offsets_mapping: bool = False, add_special_tokens: bool = False) -> dict[str, Any]:
        del add_special_tokens
        ids = [self._stoi.get(character, self.unk_token_id) for character in text]
        result: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        pieces: list[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self._itos):
                continue
            token = self._itos[token_id]
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id, self.unk_token_id}:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"type": "myna-char-tokenizer", "vocab_size": self.vocab_size}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "myna-char-tokenizer":
            raise ValueError(f"Not a Myna character tokenizer: {path}")
        return cls()
