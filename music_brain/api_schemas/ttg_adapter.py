"""
TTG v1 → engine flat contract (CompleteSongIntentRequest.structure / instruments).

PRD §8.4: v0 flat structure remains the engine boundary; TTG is adapted here.
Energy-curve-driven orchestration gating (PRD §8.2 active_threshold) is applied
when both `energy_curve` and `orchestration` are present.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from music_brain.api_schemas.ttg_v1 import (
    EnergyCurveV1,
    TTGMovementV1,
    TTGOrchestrationV1,
)


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


# ---------------------------------------------------------------------------
# Energy-curve-driven orchestration gating (PRD §8.2 active_threshold).
# ---------------------------------------------------------------------------


def interpolate_energy_at_bar(curve: EnergyCurveV1, bar: float) -> float:
    """Linearly interpolate energy at `bar`. Clamps to endpoint values outside range."""
    points = curve.points
    if bar <= points[0].bar:
        return points[0].value
    if bar >= points[-1].bar:
        return points[-1].value
    for i in range(len(points) - 1):
        lo = points[i]
        hi = points[i + 1]
        if lo.bar <= bar <= hi.bar:
            span = hi.bar - lo.bar
            if span <= 0.0:
                return lo.value
            t = (bar - lo.bar) / span
            return lo.value + t * (hi.value - lo.value)
    return points[-1].value  # unreachable; pleases the typechecker


def compute_section_energy(
    curve: EnergyCurveV1,
    section_start_bar: float,
    section_bars: int,
) -> Dict[str, float]:
    """Return {"peak": float, "mean": float} sampled across the section.

    Samples at every endpoint and at each curve breakpoint inside the section,
    so the peak is exact for piecewise-linear curves (peak always falls on a
    breakpoint or section edge).
    """
    start = float(section_start_bar)
    end = float(section_start_bar + section_bars)
    samples: List[float] = [interpolate_energy_at_bar(curve, start)]
    for point in curve.points:
        if start < point.bar < end:
            samples.append(point.value)
    samples.append(interpolate_energy_at_bar(curve, end))
    return {"peak": max(samples), "mean": sum(samples) / len(samples)}


def _structure_section_starts(structure: List[Dict[str, Any]]) -> List[float]:
    """Bar offset where each section starts, accumulated from preceding section sizes."""
    starts: List[float] = []
    cursor = 0.0
    for section in structure:
        starts.append(cursor)
        bars = int(section.get("bars", 0))
        reps = int(section.get("repetitions", 1)) or 1
        cursor += float(bars * reps)
    return starts


def orchestration_with_energy_gating(
    orch: TTGOrchestrationV1,
    energy_curve: Optional[EnergyCurveV1],
    structure: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Role map → TrackIntent dicts with per-section gating from energy_curve.

    If `energy_curve` is None or `structure` is empty, falls back to legacy
    `orchestration_to_instruments` (single ``role:`` technique, no gating).

    Otherwise, each role's `active_threshold` is compared to each section's peak
    energy. The technique list becomes ``["role:<name>", "active_in:<section>", …]``
    for sections that meet the threshold. Roles never active in any section are
    omitted from the output entirely.
    """
    if energy_curve is None or not structure:
        return orchestration_to_instruments(orch)

    section_starts = _structure_section_starts(structure)

    out: List[Dict[str, Any]] = []
    for role, spec in sorted(orch.roles.items()):
        active_sections: List[str] = []
        for section, start in zip(structure, section_starts):
            bars = int(section.get("bars", 0))
            reps = int(section.get("repetitions", 1)) or 1
            section_energy = compute_section_energy(energy_curve, start, bars * reps)
            if section_energy["peak"] >= spec.active_threshold:
                active_sections.append(str(section.get("name", "")))
        if not active_sections:
            continue
        techniques = [f"role:{role}"] + [f"active_in:{name}" for name in active_sections]
        out.append({"instrument": spec.patch, "techniques": techniques})
    return out


# ---------------------------------------------------------------------------
# Schema-expansion validation.
# ---------------------------------------------------------------------------


def validate_ttg_expansion(normalized: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable issues for a normalized TTG payload.

    Currently checks:
      - Roles whose ``active_threshold`` exceeds the max sample on ``energy_curve``
        (they would never activate, which is almost certainly a misconfiguration).

    Returns ``[]`` when energy_curve or orchestration is absent — these are
    optional fields and their absence is not an error.
    """
    issues: List[str] = []
    energy_raw = normalized.get("energy_curve")
    orch_raw = normalized.get("orchestration")
    if not energy_raw or not orch_raw:
        return issues

    points = energy_raw.get("points") if isinstance(energy_raw, dict) else None
    roles = orch_raw.get("roles") if isinstance(orch_raw, dict) else None
    if not points or not roles:
        return issues

    max_energy = max(
        float(p.get("value", 0.0)) for p in points if isinstance(p, dict)
    )
    for role_name, spec in roles.items():
        if not isinstance(spec, dict):
            continue
        threshold = float(spec.get("active_threshold", 0.0))
        if threshold > max_energy:
            issues.append(
                f"role '{role_name}' active_threshold={threshold:.2f} "
                f"exceeds max energy_curve value {max_energy:.2f}; "
                "role will never activate."
            )
    return issues
