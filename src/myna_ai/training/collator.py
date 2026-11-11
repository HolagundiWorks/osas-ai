"""Build masked labels from the OLMo 3 chat renderer."""

from __future__ import annotations

from typing import Any

from myna_ai.models.chat import render_olmo3_chat


def supervised_example(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_length: int,
) -> dict[str, list[int]] | None:
    rendered = render_olmo3_chat(messages, tools if tools else None, add_generation_prompt=False)
    encoded = tokenizer(rendered.text, return_offsets_mapping=True, add_special_tokens=False)
    if "offset_mapping" not in encoded:
        raise RuntimeError(
            "Loss masking needs offset mappings. Load a fast tokenizer with use_fast=True "
            "or use the built-in character tokenizer."
        )
    input_ids = list(encoded["input_ids"])
    labels: list[int] = []
    for token_id, (start, end) in zip(input_ids, encoded["offset_mapping"], strict=True):
        if _inside(start, end, rendered.assistant_spans):
            labels.append(int(token_id))
        else:
            labels.append(-100)
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    if not any(label != -100 for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def _inside(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def pad_batch(examples: list[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    width = max(len(example["input_ids"]) for example in examples)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    mask: list[list[int]] = []
    for example in examples:
        pad = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_id] * pad)
        labels.append(example["labels"] + [-100] * pad)
        mask.append(example["attention_mask"] + [0] * pad)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}
