"""Canonical supervised-fine-tuning conversation records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["cad", "structural", "coreops", "bim", "safety"]
ApiStatus = Literal["example_schema", "verified"]
SplitName = Literal["train", "validation", "test"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @field_validator("content")
    @classmethod
    def _content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value


class Provenance(BaseModel):
    dataset_id: str
    licence: str
    adapter: str
    adapter_version: str = "1"
    workflow_id: str | None = None
    revision: str | None = None
    notes: str = ""
    withheld_from_training: list[str] = Field(default_factory=list)


class ExpectedValidation(BaseModel):
    schema_valid: bool = True
    state_changing: bool = False
    requires_authorisation: bool = False
    must_not_invent_quantities: bool = True
    notes: str = ""


class Conversation(BaseModel):
    """One JSONL training record.

    ``api_status`` is ``example_schema`` unless a Myna application API has been
    verified. Example tool names are proposals, not confirmed product APIs.
    """

    id: str
    category: Category
    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    project_context: dict[str, Any] = Field(default_factory=dict)
    expected_validation: ExpectedValidation = Field(default_factory=ExpectedValidation)
    api_status: ApiStatus = "example_schema"
    source: Provenance
    split: SplitName | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
