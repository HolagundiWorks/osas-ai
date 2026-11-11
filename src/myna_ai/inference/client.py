"""Application-neutral inference client.

Model text is parsed and validated before any tool runs. The client has no
method that writes a drawing or a database. Cancellation and timeouts are
cooperative: the runtime must observe the cancel event or the supplied timeout.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from myna_ai import __version__
from myna_ai.common.errors import AuthorisationError, CancelledError, InferenceTimeoutError
from myna_ai.common.logging import get_logger
from myna_ai.inference.parser import parse_tool_calls
from myna_ai.models.chat import render_olmo3_chat
from myna_ai.tools.catalog import ToolCatalog, load_tool_catalog
from myna_ai.tools.mock_executor import Authorisation, ExecutionResult, MockToolExecutor, ProjectState
from myna_ai.tools.validation import validate_call

LOGGER = get_logger("inference")


class GenerationRuntime(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        cancel: threading.Event | None,
        timeout_s: float | None,
    ) -> str:
        """Return completion text. Raise CancelledError or InferenceTimeoutError."""

    def model_version(self) -> dict[str, Any]:
        """Report the loaded model without contacting the network."""


@dataclass
class InferenceRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    max_new_tokens: int = 128
    timeout_s: float | None = 30
    execute_tools: bool = False
    authorisation: Authorisation | None = None
    project: ProjectState | None = None


@dataclass
class InferenceResult:
    text: str
    tool_calls: list[dict[str, Any]]
    validation_problems: list[str]
    executions: list[ExecutionResult] = field(default_factory=list)
    model_version: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    cancelled: bool = False


class LocalInferenceClient:
    def __init__(
        self,
        runtime: GenerationRuntime,
        catalog: ToolCatalog | None = None,
        executor: MockToolExecutor | None = None,
    ) -> None:
        self.runtime = runtime
        self.catalog = catalog or load_tool_catalog()
        self.executor = executor or MockToolExecutor(self.catalog)

    def complete(self, request: InferenceRequest, cancel: threading.Event | None = None) -> InferenceResult:
        started = time.perf_counter()
        if cancel is not None and cancel.is_set():
            raise CancelledError("Inference was cancelled before generation.")
        prompt = render_olmo3_chat(
            request.messages,
            request.tools if request.tools is not None else self.catalog.public_schemas(),
            add_generation_prompt=True,
        ).text
        text = self.runtime.generate(
            prompt,
            max_new_tokens=request.max_new_tokens,
            cancel=cancel,
            timeout_s=request.timeout_s,
        )
        calls = parse_tool_calls(text)
        problems: list[str] = []
        for call in calls:
            problems.extend(f"{call['name']}: {problem}" for problem in validate_call(self.catalog, call["name"], call["arguments"]))
        executions: list[ExecutionResult] = []
        if request.execute_tools and calls and not problems:
            if request.project is None:
                raise AuthorisationError("Tool execution needs an in-memory project state. No file path is accepted.")
            for call in calls:
                spec = self.catalog.by_name(call["name"])
                if spec.state_changing and not (request.authorisation and request.authorisation.allow):
                    raise AuthorisationError(f"{call['name']} was not authorised. Project state was not changed.")
                executions.append(
                    self.executor.execute(
                        name=call["name"],
                        arguments=call["arguments"],
                        project=request.project,
                        authorisation=request.authorisation,
                    )
                )
        version = self.runtime.model_version()
        version.setdefault("myna_ai_version", __version__)
        LOGGER.info(
            "inference model=%s tools=%s valid=%s latency=%.3f",
            version.get("model_key"),
            [call["name"] for call in calls],
            not problems,
            time.perf_counter() - started,
        )
        return InferenceResult(
            text=text,
            tool_calls=calls,
            validation_problems=problems,
            executions=executions,
            model_version=version,
            latency_seconds=time.perf_counter() - started,
        )


class ScriptedRuntime:
    """Deterministic runtime for tests. It does not load weights or open a network connection."""

    def __init__(self, completion: str, model_key: str = "scripted") -> None:
        self.completion = completion
        self.model_key = model_key

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        cancel: threading.Event | None,
        timeout_s: float | None,
    ) -> str:
        del prompt, max_new_tokens
        if cancel is not None and cancel.is_set():
            raise CancelledError("Inference was cancelled.")
        if timeout_s is not None and timeout_s <= 0:
            raise InferenceTimeoutError("Inference timed out.")
        return self.completion

    def model_version(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "base_model_id": None,
            "adapter_path": None,
            "offline": True,
            "myna_ai_version": __version__,
        }
