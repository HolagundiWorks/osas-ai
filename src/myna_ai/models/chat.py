"""OLMo 3 chat rendering.

The renderer follows ``chat_template.jinja`` on
``allenai/Olmo-3-7B-Instruct`` (fetched for this repository), not the shorter
illustration in the model card. The card shows a ``<functions></functions>``
marker that the Jinja template does not emit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OLMO3_EOS = "<|endoftext|>"
_DEFAULT_PREAMBLE = "You are a helpful function-calling AI assistant. "
_NO_TOOLS = "You do not currently have access to any functions. "
_TOOLS_PREAMBLE = (
    "You are provided with function signatures within XML tags. "
    "You may call one or more functions to assist with the user query. "
    "Output any function calls within <function_calls></function_calls> XML tags. "
    "Do not make assumptions about what values to plug into functions. "
)


@dataclass
class RenderedChat:
    text: str
    assistant_spans: list[tuple[int, int]] = field(default_factory=list)


def render_olmo3_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    add_generation_prompt: bool = False,
    eos_token: str = OLMO3_EOS,
) -> RenderedChat:
    """Render messages and return character spans that belong to assistant turns."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []

    def append(text: str) -> None:
        parts.append(text)

    def position() -> int:
        return sum(len(part) for part in parts)

    has_system = any(message.get("role") == "system" for message in messages)
    if not has_system:
        append("<|im_start|>system\n" + _DEFAULT_PREAMBLE)
        if tools is None:
            append(_NO_TOOLS + "<|im_end|>\n")
        else:
            append(_TOOLS_PREAMBLE + _tools_json(tools) + " <|im_end|>\n")

    for index, message in enumerate(messages):
        role = message.get("role")
        is_last = index == len(messages) - 1
        if role == "system":
            append("<|im_start|>system\n" + str(message.get("content") or ""))
            if tools is not None:
                append(" " + _tools_json(tools) + " ")
            elif message.get("functions") is not None:
                append(" " + str(message["functions"]) + " ")
            append("<|im_end|>\n")
        elif role == "user":
            append("<|im_start|>user\n" + str(message.get("content") or "") + "<|im_end|>\n")
        elif role == "assistant":
            append("<|im_start|>assistant\n")
            start = position()
            content = message.get("content")
            if content is not None:
                append(str(content))
            calls = message.get("tool_calls") or message.get("function_calls")
            if calls:
                append(_render_function_calls(calls))
            if is_last and not add_generation_prompt:
                append(eos_token)
            else:
                append("<|im_end|>\n")
            # Train on assistant content, function calls and the turn closer.
            # The role header stays masked. Environment turns are not spans.
            spans.append((start, position()))
        elif role in {"tool", "environment"}:
            append("<|im_start|>environment\n" + str(message.get("content") or "") + "<|im_end|>\n")
        else:
            raise ValueError(f"Unsupported chat role: {role}")
        if is_last and add_generation_prompt and role != "assistant":
            append("<|im_start|>assistant\n")
    return RenderedChat(text="".join(parts), assistant_spans=spans)


def _tools_json(tools: list[dict[str, Any]]) -> str:
    import json

    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))


def _render_function_calls(calls: Any) -> str:
    """Render OLMo function calls.

    Official template shape: ``name(key="value", other=1)`` inside
    ``<function_calls>``. A string is passed through unchanged.
    """
    if isinstance(calls, str):
        body = calls
    else:
        rendered: list[str] = []
        for call in calls:
            if isinstance(call, str):
                rendered.append(call)
                continue
            function = call.get("function", call)
            name = function["name"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                rendered.append(f"{name}({arguments})")
                continue
            rendered.append(f"{name}({_format_arguments(arguments)})")
        body = "\n".join(rendered)
    return "<function_calls>" + body + "</function_calls>"


def _format_arguments(arguments: dict[str, Any]) -> str:
    import json

    pieces = [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in arguments.items()]
    return ", ".join(pieces)
