"""In-memory CAD and CoreOps stand-in.

The executor never writes project files or databases. Quantities come from the
seeded project state. Model-supplied quantities are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from myna_ai.common.errors import AuthorisationError
from myna_ai.tools.catalog import ToolCatalog, load_tool_catalog
from myna_ai.tools.validation import validate_call


@dataclass
class Authorisation:
    allow: bool
    actor: str
    reason: str = ""


@dataclass
class ProjectState:
    """Deterministic project snapshot owned by the application, not the model."""

    project_id: str
    estimate_state: str = "draft"
    revision: int = 0
    quantities: dict[str, float] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "estimate_state": self.estimate_state,
            "revision": self.revision,
            "quantities": dict(self.quantities),
            "entities": [dict(entity) for entity in self.entities],
            "history": [dict(item) for item in self.history],
        }


@dataclass
class ExecutionResult:
    ok: bool
    name: str
    output: dict[str, Any]
    state_changing: bool
    ignored_model_supplied_quantities: bool = False


class MockToolExecutor:
    def __init__(self, catalog: ToolCatalog | None = None) -> None:
        self.catalog = catalog or load_tool_catalog()

    def execute(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        project: ProjectState,
        authorisation: Authorisation | None = None,
    ) -> ExecutionResult:
        problems = validate_call(self.catalog, name, arguments)
        if problems:
            return ExecutionResult(
                ok=False,
                name=name,
                output={"error": "schema_invalid", "problems": problems},
                state_changing=False,
            )
        spec = self.catalog.by_name(name)
        if spec.state_changing and not (authorisation and authorisation.allow):
            raise AuthorisationError(
                f"{name} changes project state and needs an explicit authorisation decision."
            )
        before = project.snapshot()
        ignored = _contains_quantity(arguments)
        output = _dispatch(name, arguments, project, ignored)
        if spec.state_changing:
            project.history.append(
                {
                    "tool": name,
                    "actor": None if authorisation is None else authorisation.actor,
                    "before_revision": before["revision"],
                    "after_revision": project.revision,
                    "estimate_state": project.estimate_state,
                }
            )
        return ExecutionResult(
            ok=True,
            name=name,
            output=output,
            state_changing=spec.state_changing,
            ignored_model_supplied_quantities=ignored and name.startswith("coreops."),
        )


def _contains_quantity(arguments: dict[str, Any]) -> bool:
    return any(key in arguments for key in ("quantity", "quantities", "volume", "cost", "weight"))


def _dispatch(name: str, arguments: dict[str, Any], project: ProjectState, ignored: bool) -> dict[str, Any]:
    if name == "cad.select_entities":
        layer = (arguments.get("criteria") or {}).get("layer")
        floor = (arguments.get("criteria") or {}).get("floor")
        matched = []
        for entity in project.entities:
            if layer is not None and entity.get("layer") != layer:
                continue
            if floor is not None and entity.get("floor") != floor:
                continue
            matched.append(dict(entity))
        return {"entities": matched, "count": len(matched)}
    if name == "cad.create_geometry":
        entity = {
            "entity_id": f"ent-{len(project.entities) + 1}",
            "kind": arguments["kind"],
            "layer": arguments["layer"],
            "geometry": arguments["geometry"],
            "units": arguments["units"],
        }
        project.entities.append(entity)
        return {"created": entity, "persisted": False, "note": "Mock only. No drawing file was written."}
    if name == "cad.modify_geometry":
        target = arguments["entity_id"]
        found = next((entity for entity in project.entities if entity.get("entity_id") == target), None)
        if found is None:
            return {"updated": False, "reason": "entity_not_in_project_state"}
        found["geometry"] = arguments["geometry"]
        return {"updated": True, "entity_id": target, "persisted": False}
    if name == "cad.filter_view":
        return {
            "layers": arguments.get("layers") or [],
            "floors": arguments.get("floors") or [],
            "applied_to_file": False,
        }
    if name == "structural.describe_element":
        return {
            "element_id": arguments["element_id"],
            "element_type": arguments["element_type"],
            "properties_recorded": arguments.get("properties") or {},
            "design_check": None,
            "note": "No structural design result was calculated.",
        }
    if name == "coreops.extract_quantities":
        return {
            "project_id": project.project_id,
            "quantities": dict(project.quantities),
            "source": "project_state",
            "ignored_model_supplied_quantities": ignored,
        }
    if name == "coreops.update_draft_estimate":
        if project.estimate_state != "draft":
            return {
                "updated": False,
                "reason": "estimate_is_not_draft",
                "estimate_state": project.estimate_state,
                "note": "Approved estimates change through coreops.revise_approved_estimate.",
            }
        project.revision += 1
        return {
            "updated": True,
            "estimate_state": project.estimate_state,
            "revision": project.revision,
            "quantities": dict(project.quantities),
            "ignored_model_supplied_quantities": ignored,
            "persisted": False,
        }
    if name == "coreops.revise_approved_estimate":
        if project.estimate_state != "approved":
            return {
                "revised": False,
                "reason": "estimate_is_not_approved",
                "estimate_state": project.estimate_state,
            }
        reason = str(arguments.get("reason") or "")
        project.revision += 1
        project.history.append({"kind": "approved_revision", "reason": reason, "revision": project.revision})
        return {
            "revised": True,
            "estimate_state": "approved",
            "revision": project.revision,
            "quantities": dict(project.quantities),
            "ignored_model_supplied_quantities": ignored,
            "persisted": False,
            "note": "Revision counter is deterministic application state.",
        }
    if name == "bim.identify_elements":
        return {
            "candidates": [
                {"entity_id": entity.get("entity_id"), "bim_concept": "unclassified", "complete": False}
                for entity in project.entities
            ],
            "missing_information": ["material", "structural_function"],
            "model_generated": False,
        }
    if name == "bim.propose_model_generation":
        return {
            "accepted": False,
            "reason": arguments.get("reason") or "proposal_only",
            "model_generated": False,
            "note": "A proposal does not create a BIM model or quantities.",
        }
    return {"ok": False, "reason": "no_dispatcher", "name": name}
