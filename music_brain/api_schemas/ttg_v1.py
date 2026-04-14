"""
Temporal Tension Graph (TTG) v1 — PRD §8 (MVP subset).

Movements → phrases; optional boundary_event drum_fill_1bar; role orchestration
with energy activation thresholds; optional energy curve samples.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SECTION_NAME_PATTERN = r"^(intro|verse|chorus|bridge|outro|build|drop)$"

BoundaryEventV1 = Literal["drum_fill_1bar"]
HarmonicRhythmV1 = Literal["slow", "fast", "medium"]


class TTGPhraseV1(BaseModel):
    """Meso-level phrase inside a movement (PRD §8.2–8.3)."""

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = Field(default=None, max_length=64)
    bars: int = Field(ge=1, le=128)
    harmonic_rhythm: Optional[HarmonicRhythmV1] = None
    motifs: List[str] = Field(default_factory=list, max_length=32)
    boundary_event: Optional[BoundaryEventV1] = None
    section_role: Optional[str] = Field(
        default=None,
        description="Maps to engine flat section name; if omitted, inferred from phrase index.",
        pattern=SECTION_NAME_PATTERN,
    )


class TTGMovementV1(BaseModel):
    """Macro movement: root of `technical.timeline` (PRD §8.3)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["movement"] = "movement"
    id: str = Field(min_length=1, max_length=64)
    bars: Optional[int] = Field(
        default=None,
        ge=1,
        le=1024,
        description="Optional; when set, must match derived bar count from children + fills.",
    )
    children: List[TTGPhraseV1] = Field(min_length=1)

    @model_validator(mode="after")
    def bars_matches_children(self) -> "TTGMovementV1":
        if self.bars is None:
            return self
        derived = _derived_movement_bars(self.children)
        if self.bars != derived:
            raise ValueError(
                f"timeline.bars ({self.bars}) does not match derived bar count ({derived}) "
                "from phrases and boundary_event fills"
            )
        return self


def _derived_movement_bars(children: List[TTGPhraseV1]) -> int:
    total = 0
    for phrase in children:
        total += phrase.bars
        if phrase.boundary_event == "drum_fill_1bar":
            total += 1
    return total


class OrchestrationRoleSpecV1(BaseModel):
    """Role → patch + energy gate (PRD §8.2, §8.3 orchestration)."""

    model_config = ConfigDict(extra="forbid")

    patch: str = Field(min_length=1, max_length=128)
    active_threshold: float = Field(ge=0.0, le=1.0)


class TTGOrchestrationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: Dict[str, OrchestrationRoleSpecV1] = Field(min_length=1)


class EnergyCurvePointV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bar: float = Field(ge=0.0, description="Musical time anchor in bars (fractional allowed).")
    value: float = Field(ge=0.0, le=1.0)


class EnergyCurveV1(BaseModel):
    """First-class energy samples over musical time (PRD §8.2); optional for MVP."""

    model_config = ConfigDict(extra="forbid")

    points: List[EnergyCurvePointV1] = Field(min_length=2)
