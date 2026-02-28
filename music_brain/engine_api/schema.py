"""Strict schema validation at the UI-to-engine boundary."""

from __future__ import annotations

from typing import List, Optional

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, validator as field_validator  # type: ignore


class TrackIntent(BaseModel):
    instrument: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Canonical instrument name (e.g., piano, synth_bass)",
    )
    techniques: List[str] = Field(default_factory=list, description="Allowed techniques")


class StructureSection(BaseModel):
    name: str = Field(..., pattern=r"^(intro|verse|chorus|bridge|outro|build|drop)$")
    bars: int = Field(ge=1, le=128)
    repetitions: int = Field(default=1, ge=1, le=16)


class CompleteSongIntentRequest(BaseModel):
    core_desire: str = Field(..., min_length=1, max_length=1000)
    mood_primary: str = Field(..., min_length=1, max_length=100)
    genre: str = Field(..., min_length=1, max_length=100)
    tempo: int = Field(default=120, ge=40, le=300, description="BPM clamped to engine limits")
    key_mode: str = Field(
        ...,
        pattern=r"^[A-G][#b]?\s(major|minor|dorian|mixolydian|lydian|phrygian|locrian)$",
    )
    structure: List[StructureSection] = Field(min_length=1)
    instruments: List[TrackIntent] = Field(min_length=1)
    allow_legacy_fallback: bool = Field(default=False)

    # UI-sourced musical constraints (no longer silently dropped)
    groove_feel: str = Field(
        default="Straight/Driving",
        description="Rhythmic feel dictating quantization/swing",
    )
    narrative_arc: str = Field(
        default="Climb-to-Climax",
        description="Overall energetic trajectory of the song",
    )
    rule_to_break: Optional[str] = Field(
        default=None,
        description="Intentional music theory violation for emotional effect",
    )
    rule_justification: Optional[str] = Field(
        default=None,
        description="Narrative reason for the rule break",
    )

    @field_validator("structure")
    @classmethod
    def validate_total_duration(cls, sections: List[StructureSection]) -> List[StructureSection]:
        total_bars = sum(section.bars * section.repetitions for section in sections)
        if total_bars > 1000:
            raise ValueError(f"Total structure exceeds maximum safe bar count: {total_bars}")
        return sections

