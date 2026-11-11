"""Validate canonical conversations before they are written to a split."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from myna_ai.datasets.schema import Conversation
from myna_ai.tools.catalog import tool_names


class RecordProblem(Exception):
    def __init__(self, record_id: str, messages: list[str]) -> None:
        self.record_id = record_id
        self.messages = messages
        super().__init__(f"{record_id}: {'; '.join(messages)}")


def validate_conversation(payload: dict[str, Any], *, known_tools: set[str] | None = None) -> Conversation:
    problems: list[str] = []
    record_id = str(payload.get("id") or "<missing-id>")
    try:
        conversation = Conversation.model_validate(payload)
    except ValidationError as exc:
        raise RecordProblem(record_id, [err["msg"] for err in exc.errors()]) from exc

    roles = [message.role for message in conversation.messages]
    if not roles or roles[0] != "system":
        problems.append("The first message must be a system message.")
    if "user" not in roles or "assistant" not in roles:
        problems.append("A conversation needs at least one user message and one assistant message.")
    if conversation.api_status == "verified":
        problems.append(
            "api_status 'verified' is reserved for a checked Myna application API. "
            "No production API mapping is registered in this repository."
        )
    allowed = set(known_tools or tool_names())
    declared = set()
    for tool in conversation.tools:
        name = tool.get("name")
        if isinstance(name, str):
            declared.add(name)
            if name not in allowed:
                problems.append(f"Tool '{name}' is not in the proposed tool catalog.")
    pending: dict[str, str] = {}
    for message in conversation.messages:
        if message.role == "assistant" and message.tool_calls:
            if not message.tool_calls and not message.content:
                problems.append("Assistant message is empty.")
            for call in message.tool_calls:
                if call.name not in allowed:
                    problems.append(f"Tool call '{call.name}' is not in the proposed tool catalog.")
                if declared and call.name not in declared:
                    problems.append(f"Tool call '{call.name}' is not declared on this conversation.")
                pending[call.id] = call.name
        if message.role == "tool":
            if not message.tool_call_id or message.tool_call_id not in pending:
                problems.append("Tool message does not match a preceding assistant tool call.")
            elif message.tool_call_id in pending:
                del pending[message.tool_call_id]
        if message.role == "assistant" and not message.tool_calls and not (message.content and message.content.strip()):
            problems.append("Assistant message has no content.")
    if pending:
        problems.append("Assistant tool calls are missing tool results: " + ", ".join(sorted(pending)))
    if conversation.category == "safety":
        assistant_text = " ".join(
            message.content or "" for message in conversation.messages if message.role == "assistant"
        ).lower()
        if not any(token in assistant_text for token in ("cannot", "missing", "ambiguous", "unsupported", "not ")):
            problems.append("Safety examples must state a refusal, a missing input, or an unsupported operation.")
    if problems:
        raise RecordProblem(conversation.id, problems)
    return conversation


def validate_records(records: list[dict[str, Any]]) -> tuple[list[Conversation], list[dict[str, Any]]]:
    accepted: list[Conversation] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        record_id = str(record.get("id") or "")
        if record_id in seen:
            rejected.append({"id": record_id, "problems": ["Duplicate conversation id."]})
            continue
        seen.add(record_id)
        try:
            accepted.append(validate_conversation(record))
        except RecordProblem as exc:
            rejected.append({"id": exc.record_id, "problems": exc.messages})
    return accepted, rejected
