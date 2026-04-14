"""
TTG v1 → engine flat contract (CompleteSongIntentRequest.structure / instruments).

PRD §8.4: v0 flat structure remains the engine boundary; TTG is adapted here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from music_brain.api_schemas.ttg_v1 import TTGMovementV1, TTGOrchestrationV1


def infer_section_role(index: int, total: int) -> str:
    """Deterministic section labels when phrase.section_role is omitted."""
    if total <= 0:
        return "verse"
    if total == 1:
        return "verse"
    if total == 2:
        return "intro" if index == 0 else "chorus"
    if index == 0:
        return "intro"
    if index == total - 1:
        return "outro"
    return "verse" if index % 2 == 1 else "chorus"


def ttg_movement_to_structure_sections(movement: TTGMovementV1) -> List[Dict[str, Any]]:
    """Flatten movement+phrases (+ drum fill bars) to StructureSection-shaped dicts."""
    children = movement.children
    total = len(children)
    out: List[Dict[str, Any]] = []
    for i, phrase in enumerate(children):
        name = phrase.section_role or infer_section_role(i, total)
        out.append({"name": name, "bars": phrase.bars, "repetitions": 1})
        if phrase.boundary_event == "drum_fill_1bar":
            out.append({"name": "build", "bars": 1, "repetitions": 1})
    return out


def orchestration_to_instruments(orch: TTGOrchestrationV1) -> List[Dict[str, Any]]:
    """Role map → TrackIntent-shaped dicts (patch names as instrument ids)."""
    return [
        {"instrument": spec.patch, "techniques": [f"role:{role}"]}
        for role, spec in sorted(orch.roles.items())
    ]


def ttg_dict_to_structure_list(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse and flatten a raw timeline dict."""
    movement = TTGMovementV1.model_validate(timeline)
    return ttg_movement_to_structure_sections(movement)


def orchestration_dict_to_instruments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    orch = TTGOrchestrationV1.model_validate(data)
    return orchestration_to_instruments(orch)
