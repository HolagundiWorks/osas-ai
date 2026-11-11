import threading
from pathlib import Path

import pytest

from myna_ai.common.errors import AuthorisationError, CancelledError
from myna_ai.inference.client import InferenceRequest, LocalInferenceClient, ScriptedRuntime
from myna_ai.inference.parser import parse_tool_calls
from myna_ai.tools.mock_executor import Authorisation, MockToolExecutor, ProjectState


def test_parser_reads_olmo_function_calls() -> None:
    text = (
        '<function_calls>cad.create_geometry(project_id="tower-a", kind="line", '
        'layer="A-GRID", units="mm", geometry={"start": [0, 0], "end": [4500, 0]}'
        ")</function_calls>"
    )
    calls = parse_tool_calls(text)
    assert calls[0]["name"] == "cad.create_geometry"
    assert calls[0]["arguments"]["geometry"]["end"] == [4500, 0]


def test_state_change_requires_authorisation_and_does_not_write_files(tmp_path: Path) -> None:
    project = ProjectState(project_id="synthetic-project", quantities={"concrete_m3": 12.5})
    client = LocalInferenceClient(
        ScriptedRuntime(
            '<function_calls>cad.create_geometry(project_id="synthetic-project", kind="line", '
            'layer="A-GRID", units="mm", geometry={"start": [0, 0], "end": [6000, 0]})</function_calls>'
        )
    )
    request = InferenceRequest(
        messages=[{"role": "user", "content": "Create the line."}],
        execute_tools=True,
        project=project,
    )
    with pytest.raises(AuthorisationError):
        client.complete(request)
    assert project.entities == []
    allowed = client.complete(
        InferenceRequest(
            messages=request.messages,
            execute_tools=True,
            project=project,
            authorisation=Authorisation(allow=True, actor="tester", reason="fixture"),
        )
    )
    assert allowed.executions[0].output["persisted"] is False
    assert project.entities[0]["entity_id"] == "ent-1"
    assert list(tmp_path.iterdir()) == []


def test_quantities_come_from_project_state() -> None:
    project = ProjectState(project_id="synthetic-project", quantities={"concrete_m3": 12.5})
    result = MockToolExecutor().execute(
        name="coreops.extract_quantities",
        arguments={"project_id": "synthetic-project"},
        project=project,
        authorisation=None,
    )
    assert result.output["quantities"] == {"concrete_m3": 12.5}


def test_approved_estimate_is_not_edited_as_a_draft() -> None:
    project = ProjectState(project_id="synthetic-project", estimate_state="approved", revision=4)
    auth = Authorisation(allow=True, actor="tester")
    draft = MockToolExecutor().execute(
        name="coreops.update_draft_estimate",
        arguments={"project_id": "synthetic-project", "reason": "bypass"},
        project=project,
        authorisation=auth,
    )
    assert draft.output["updated"] is False
    assert project.revision == 4
    revised = MockToolExecutor().execute(
        name="coreops.revise_approved_estimate",
        arguments={"project_id": "synthetic-project", "reason": "quantity changed"},
        project=project,
        authorisation=auth,
    )
    assert revised.output["revision"] == 5
    assert project.estimate_state == "approved"


def test_cancellation() -> None:
    cancel = threading.Event()
    cancel.set()
    client = LocalInferenceClient(ScriptedRuntime("unused"))
    with pytest.raises(CancelledError):
        client.complete(InferenceRequest(messages=[{"role": "user", "content": "Stop."}]), cancel=cancel)
