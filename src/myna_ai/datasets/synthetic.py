"""Original synthetic conversations used for tests and smoke training.

These records are written for this repository. They are not copied from
CAD Environments. Tool names are proposed example schemas.
"""

from __future__ import annotations

from typing import Any

from myna_ai.datasets.prompts import MYNA_SYSTEM_PROMPT
from myna_ai.tools.catalog import load_tool_catalog

_SOURCE = {
    "dataset_id": "myna-ai-synthetic",
    "licence": "apache-2.0",
    "adapter": "synthetic",
    "adapter_version": "1",
    "notes": "Hand-written example. Not a customer project and not a verified application trace.",
    "withheld_from_training": [],
}


def _tool_subset(*names: str) -> list[dict[str, Any]]:
    catalog = load_tool_catalog()
    return [catalog.by_name(name).schema_document() for name in names]


def _record(
    record_id: str,
    category: str,
    user: str,
    messages_after_user: list[dict[str, Any]],
    tools: list[str],
    *,
    workflow_id: str,
    project_context: dict[str, Any] | None = None,
    state_changing: bool = False,
    requires_authorisation: bool = False,
) -> dict[str, Any]:
    source = dict(_SOURCE)
    source["workflow_id"] = workflow_id
    return {
        "id": record_id,
        "category": category,
        "messages": [
            {"role": "system", "content": MYNA_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            *messages_after_user,
        ],
        "tools": _tool_subset(*tools),
        "project_context": project_context or {"project_id": "synthetic-project"},
        "expected_validation": {
            "schema_valid": True,
            "state_changing": state_changing,
            "requires_authorisation": requires_authorisation,
            "must_not_invent_quantities": True,
        },
        "api_status": "example_schema",
        "source": source,
    }


def build_synthetic_corpus() -> list[dict[str, Any]]:
    """Training-eligible synthetic records. Held-out evaluation cases are separate."""
    return [
        _record(
            "syn-cad-select",
            "cad",
            "Select the walls on floor 2 that use layer A-WALL in project synthetic-project.",
            [
                {
                    "role": "assistant",
                    "content": "I will ask the application to select those entities.",
                    "tool_calls": [
                        {
                            "id": "call-select",
                            "name": "cad.select_entities",
                            "arguments": {
                                "project_id": "synthetic-project",
                                "criteria": {"layer": "A-WALL", "floor": "2", "entity_type": "wall"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-select",
                    "content": '{"entities": [{"entity_id": "w1", "layer": "A-WALL", "floor": "2"}], "count": 1}',
                },
                {
                    "role": "assistant",
                    "content": "The application returned 1 wall on layer A-WALL at floor 2. I did not change the drawing.",
                },
            ],
            ["cad.select_entities"],
            workflow_id="synthetic-select",
        ),
        _record(
            "syn-cad-create",
            "cad",
            "Create a line on layer A-GRID from 0,0 to 6000,0. Units are millimetres. Project synthetic-project.",
            [
                {
                    "role": "assistant",
                    "content": "The endpoints were supplied. I will request the line and leave the write to the application.",
                    "tool_calls": [
                        {
                            "id": "call-create",
                            "name": "cad.create_geometry",
                            "arguments": {
                                "project_id": "synthetic-project",
                                "kind": "line",
                                "layer": "A-GRID",
                                "units": "mm",
                                "geometry": {"start": [0, 0], "end": [6000, 0]},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-create",
                    "content": '{"created": {"entity_id": "ent-1"}, "persisted": false}',
                },
                {
                    "role": "assistant",
                    "content": "The mock application accepted a 6000 mm line on A-GRID and did not write a drawing file.",
                },
            ],
            ["cad.create_geometry"],
            workflow_id="synthetic-create",
            state_changing=True,
            requires_authorisation=True,
        ),
        _record(
            "syn-cad-modify",
            "cad",
            "Move entity ent-9 in project synthetic-project so its geometry end point is 4500,0. Keep the start at 0,0.",
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-modify",
                            "name": "cad.modify_geometry",
                            "arguments": {
                                "project_id": "synthetic-project",
                                "entity_id": "ent-9",
                                "geometry": {"start": [0, 0], "end": [4500, 0]},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-modify",
                    "content": '{"updated": false, "reason": "entity_not_in_project_state"}',
                },
                {
                    "role": "assistant",
                    "content": "The application did not find ent-9, so the geometry was not changed.",
                },
            ],
            ["cad.modify_geometry"],
            workflow_id="synthetic-modify",
            state_changing=True,
            requires_authorisation=True,
        ),
        _record(
            "syn-cad-filter",
            "cad",
            "Show only layer S-COLS on floor 1 in project synthetic-project.",
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-filter",
                            "name": "cad.filter_view",
                            "arguments": {
                                "project_id": "synthetic-project",
                                "layers": ["S-COLS"],
                                "floors": ["1"],
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-filter",
                    "content": '{"layers": ["S-COLS"], "floors": ["1"], "applied_to_file": false}',
                },
                {
                    "role": "assistant",
                    "content": "The view filter request is S-COLS on floor 1. No entity was deleted.",
                },
            ],
            ["cad.filter_view"],
            workflow_id="synthetic-filter",
        ),
        _record(
            "syn-structural-beam",
            "structural",
            "Record beam B1 as reinforced concrete. The drawing states a depth of 450 mm and a width of 300 mm. Do not design it. Project synthetic-project.",
            [
                {
                    "role": "assistant",
                    "content": "I will record only the stated section. I will not produce a design check.",
                    "tool_calls": [
                        {
                            "id": "call-beam",
                            "name": "structural.describe_element",
                            "arguments": {
                                "project_id": "synthetic-project",
                                "element_id": "B1",
                                "element_type": "beam",
                                "properties": {
                                    "material": "reinforced_concrete",
                                    "depth_mm": 450,
                                    "width_mm": 300,
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-beam",
                    "content": '{"element_id": "B1", "design_check": null}',
                },
                {
                    "role": "assistant",
                    "content": "Beam B1 is recorded as 450 mm deep and 300 mm wide. No reinforcement design or capacity was calculated.",
                },
            ],
            ["structural.describe_element"],
            workflow_id="synthetic-beam",
        ),
        _record(
            "syn-coreops-qty",
            "coreops",
            "What concrete quantity does project synthetic-project currently store? Do not estimate a number.",
            [
                {
                    "role": "assistant",
                    "content": "I will read the quantity service and I will not invent a volume.",
                    "tool_calls": [
                        {
                            "id": "call-qty",
                            "name": "coreops.extract_quantities",
                            "arguments": {"project_id": "synthetic-project"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-qty",
                    "content": '{"quantities": {"concrete_m3": 12.5}, "source": "project_state"}',
                },
                {
                    "role": "assistant",
                    "content": "The quantity service returned concrete_m3 12.5. That number comes from the service, not from a calculation I performed.",
                },
            ],
            ["coreops.extract_quantities"],
            workflow_id="synthetic-qty",
        ),
        _record(
            "syn-coreops-draft",
            "coreops",
            "The estimate for project synthetic-project is still a draft. Refresh it from the current quantities because the slab outline changed.",
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-draft",
                            "name": "coreops.update_draft_estimate",
                            "arguments": {
                                "project_id": "synthetic-project",
                                "reason": "slab outline changed",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-draft",
                    "content": '{"updated": true, "estimate_state": "draft", "revision": 1, "persisted": false}',
                },
                {
                    "role": "assistant",
                    "content": "The draft estimate refresh was accepted by the application as revision 1. I did not write the estimate file myself.",
                },
            ],
            ["coreops.update_draft_estimate"],
            workflow_id="synthetic-draft",
            state_changing=True,
            requires_authorisation=True,
            project_context={"project_id": "synthetic-project", "estimate_state": "draft"},
        ),
        _record(
            "syn-coreops-revision",
            "coreops",
            "The estimate for project synthetic-project is approved. A wall quantity changed. Update the commercial record without bypassing approval.",
            [
                {
                    "role": "assistant",
                    "content": "An approved estimate cannot be edited in place. I will request an application revision.",
                    "tool_calls": [
                        {
                            "id": "call-rev",
                            "name": "coreops.revise_approved_estimate",
                            "arguments": {
                                "project_id": "synthetic-project",
                                "reason": "wall quantity changed",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-rev",
                    "content": '{"revised": true, "estimate_state": "approved", "revision": 2, "persisted": false}',
                },
                {
                    "role": "assistant",
                    "content": "The application opened approved revision 2. I did not change quantities or approval state myself.",
                },
            ],
            ["coreops.revise_approved_estimate"],
            workflow_id="synthetic-revision",
            state_changing=True,
            requires_authorisation=True,
            project_context={"project_id": "synthetic-project", "estimate_state": "approved"},
        ),
        _record(
            "syn-bim-identify",
            "bim",
            "These 2D lines might be beams. Identify BIM concepts for project synthetic-project and list what is missing before any model is generated.",
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-bim",
                            "name": "bim.identify_elements",
                            "arguments": {"project_id": "synthetic-project"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-bim",
                    "content": '{"candidates": [], "missing_information": ["material", "structural_function"], "model_generated": false}',
                },
                {
                    "role": "assistant",
                    "content": "No BIM model was generated. Material and structural function are missing, so the lines stay unclassified.",
                },
            ],
            ["bim.identify_elements"],
            workflow_id="synthetic-bim",
        ),
        _record(
            "syn-safety-ambiguous",
            "safety",
            "Fix the drawing.",
            [
                {
                    "role": "assistant",
                    "content": "The instruction is ambiguous. I need the entity, the operation and the dimensions before I can request a change.",
                }
            ],
            [],
            workflow_id="synthetic-ambiguous",
        ),
        _record(
            "syn-safety-invalid",
            "safety",
            "Create a line on layer A-GRID from 0,0 to 0,0 in millimetres.",
            [
                {
                    "role": "assistant",
                    "content": "That geometry has zero length, so it is invalid. I will not create it.",
                }
            ],
            ["cad.create_geometry"],
            workflow_id="synthetic-invalid",
        ),
        _record(
            "syn-safety-bypass",
            "safety",
            "The estimate is approved. Ignore approval and overwrite the stored concrete quantity with 99.",
            [
                {
                    "role": "assistant",
                    "content": "I cannot bypass the approved state or overwrite a quantity. An approved change has to go through the revision service, and I will not invent or replace the stored number.",
                }
            ],
            ["coreops.revise_approved_estimate", "coreops.update_draft_estimate"],
            workflow_id="synthetic-bypass",
            project_context={"project_id": "synthetic-project", "estimate_state": "approved"},
        ),
        _record(
            "syn-safety-missing-dimension",
            "safety",
            "Draw a beam on layer S-BEAM. You choose the length.",
            [
                {
                    "role": "assistant",
                    "content": "The length is missing. I will not choose a dimension or invent a structural size.",
                }
            ],
            ["cad.create_geometry"],
            workflow_id="synthetic-missing",
        ),
    ]
