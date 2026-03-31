"""Strict schema validation at the UI-to-engine boundary."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, validator as field_validator  # type: ignore


class EmotionTag(str, Enum):
    TENSION = "tension"
    RELEASE = "release"
    WARM = "warm"
    COLD = "cold"
    BRIGHT = "bright"
    DARK = "dark"
    DRIVE = "drive"
    FLOAT = "float"


class EmotionStateSchema(BaseModel):
    """Canonical emotion contract v1. Source of truth for all language bindings."""
    model_config = {"extra": "forbid"}

    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Negative to positive [-1, 1]")
    arousal: float = Field(default=0.5, ge=0.0, le=1.0, description="Calm to excited [0, 1]")
    dominance: float = Field(default=0.5, ge=0.0, le=1.0, description="Submissive to dominant [0, 1]")
    tags: List[EmotionTag] = Field(
        default_factory=list,
        max_length=3,
        description="Max 3 tags from controlled vocabulary",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Inference quality gate [0, 1]")

    @field_validator("tags")
    @classmethod
    def validate_unique_tags(cls, v: List[EmotionTag]) -> List[EmotionTag]:
        if len(v) != len(set(v)):
            raise ValueError("Tags must be unique")
        return v


class IntentMetaSchema(BaseModel):
    """Intent metadata — version and routing IDs."""
    model_config = {"extra": "forbid"}
    schema_version: int = Field(default=1, description="Schema version")
    intent_id: int = Field(default=0, ge=0, description="Monotonic intent ID")
    session_id: int = Field(default=0, ge=0, description="Session ID")

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported schema version: {v}")
        return v


class MusicalIntentSchema(BaseModel):
    """Musical intent — biases and tendencies."""
    model_config = {"extra": "forbid"}
    tempo_bias: float = Field(default=0.0, ge=-1.0, le=1.0)
    rhythmic_density: float = Field(default=0.5, ge=0.0, le=1.0)
    groove_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    harmonic_tension: float = Field(default=0.5, ge=0.0, le=1.0)
    harmonic_motion: float = Field(default=0.5, ge=0.0, le=1.0)
    mode_preference: int = Field(default=0, ge=-1, le=1)
    melodic_activity: float = Field(default=0.5, ge=0.0, le=1.0)
    contour_variance: float = Field(default=0.5, ge=0.0, le=1.0)
    dynamic_range: float = Field(default=0.5, ge=0.0, le=1.0)
    texture_density: float = Field(default=0.5, ge=0.0, le=1.0)


class SectionRole(str, Enum):
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    BUILD = "build"
    DROP = "drop"
    UNSPECIFIED = ""


class MusicHintsSchema(BaseModel):
    """Music hints — key, tempo, chord bias, section role."""
    model_config = {"extra": "forbid"}
    key: str = Field(default="", max_length=3, description="Key (e.g. 'C', 'F#')")
    tempo_bpm: float = Field(default=0.0, ge=0.0, description="Tempo BPM (0 = unspecified)")
    chord_bias: str = Field(default="", max_length=32, description="Chord bias")
    section_role: SectionRole = Field(
        default=SectionRole.UNSPECIFIED, description="Section role"
    )


class DSPTargetsSchema(BaseModel):
    """DSP targets with per-parameter confidence and stale flag.
    Safe defaults: filter mid-open, reverb subtle, drive off, stale=True.
    """
    model_config = {"extra": "forbid"}
    filter_cutoff: float = Field(default=0.5, ge=0.0, le=1.0)
    filter_cutoff_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reverb_send: float = Field(default=0.2, ge=0.0, le=1.0)
    reverb_send_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    drive: float = Field(default=0.0, ge=0.0, le=1.0)
    drive_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    stale: bool = Field(default=True, description="True if DSP values are not yet valid")


class TimeScopeSchema(BaseModel):
    """Time scope — intent without time is noise."""
    model_config = {"extra": "forbid"}
    start_bar: int = Field(default=-1, description="Start bar (-1 = immediate)")
    end_bar: int = Field(default=-1, description="End bar (-1 = open-ended)")
    fade_in_beats: float = Field(default=0.0, ge=0.0)
    fade_out_beats: float = Field(default=0.0, ge=0.0)

    @field_validator("end_bar")
    @classmethod
    def validate_time_scope(cls, end_bar: int, info) -> int:
        start_bar = info.data.get("start_bar", -1)
        if end_bar != -1 and start_bar != -1 and end_bar <= start_bar:
            raise ValueError(
                f"end_bar ({end_bar}) must be > start_bar ({start_bar})"
            )
        return end_bar


class IntentConstraintsSchema(BaseModel):
    """Intent constraints — limit generation, not force it."""
    model_config = {"extra": "forbid"}
    allowed_engines_mask: int = Field(default=0xFFFFFFFF, ge=0)
    forbidden_engines_mask: int = Field(default=0, ge=0)
    max_cpu_cost: float = Field(default=1.0, ge=0.0)
    max_event_rate: float = Field(default=1000.0, ge=0.0)


class IntentProvenanceSchema(BaseModel):
    """Intent provenance — debugging and trust."""
    model_config = {"extra": "forbid"}
    source: int = Field(default=0, ge=0, le=5)
    user_override_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class IntentFrameSchema(BaseModel):
    """IntentFrame — top-level unit representing one musical intention."""
    model_config = {"extra": "forbid"}
    meta: IntentMetaSchema = Field(default_factory=IntentMetaSchema)
    timestamp_ms: int = Field(
        default=0, ge=0, description="Monotonic ms since session start"
    )
    emotion: EmotionStateSchema = Field(default_factory=EmotionStateSchema)
    music: MusicalIntentSchema = Field(default_factory=MusicalIntentSchema)
    music_hints: MusicHintsSchema = Field(default_factory=MusicHintsSchema)
    dsp_targets: DSPTargetsSchema = Field(default_factory=DSPTargetsSchema)
    time: TimeScopeSchema = Field(default_factory=TimeScopeSchema)
    constraints: IntentConstraintsSchema = Field(
        default_factory=IntentConstraintsSchema
    )
    provenance: IntentProvenanceSchema = Field(
        default_factory=IntentProvenanceSchema
    )
    latency_budget_ms: float = Field(
        default=10.0, ge=0.0, description="Max ms for RT engine"
    )


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
        pattern=r"^[A-G][#b]?\s(major|minor|dorian|mixolydian|lydian|phrygian|aeolian|locrian)$",
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
