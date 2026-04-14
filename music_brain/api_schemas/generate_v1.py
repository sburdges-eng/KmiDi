"""
PRD-aligned generate API models (intent_schema_version, TTG v1 on technical).

See docs/prd/KMIDI_PRD.md §6–8, §7 versioning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from music_brain.api_schemas.ttg_v1 import (
    EnergyCurveV1,
    TTGMovementV1,
    TTGOrchestrationV1,
)


class TechnicalIntent(BaseModel):
    """Musical/production parameters. Use `timeline` (TTG v1) and/or flat `structure` (v0)."""

    key: Optional[str] = None
    bpm: Optional[int] = None
    progression: Optional[List[str]] = None
    genre: Optional[str] = None
    duration: Optional[float] = None
    structure: Optional[List[Dict[str, Any]]] = None
    instruments: Optional[List[Dict[str, Any]]] = None
    techniques: Optional[List[str]] = None
    groove_feel: Optional[str] = None
    narrative_arc: Optional[str] = None
    rule_to_break: Optional[str] = None
    rule_justification: Optional[str] = None

    # --- PRD TTG v1 (docs/prd/KMIDI_PRD.md §8) ---
    timeline: Optional[TTGMovementV1] = Field(
        default=None,
        description="Temporal Tension Graph root movement; takes precedence over flat structure.",
    )
    orchestration: Optional[TTGOrchestrationV1] = Field(
        default=None,
        description="Role → patch + active_threshold; when set, drives instruments list.",
    )
    energy_curve: Optional[EnergyCurveV1] = Field(
        default=None,
        description="Optional energy samples over bars; forwarded in normalize metadata.",
    )


class EmotionalIntent(BaseModel):
    core_wound: Optional[str] = None
    core_desire: Optional[str] = None
    emotional_intent: str
    technical: Optional[TechnicalIntent] = None
    vulnerability_scale: Optional[float] = None
    narrative_arc: Optional[str] = None
    imagery_texture: Optional[str] = None


# OpenAPI / Swagger: multi-example selector on POST /generate and POST /v1/generate.
GENERATE_REQUEST_OPENAPI_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "flat_v0": {
        "summary": "Flat structure (v0)",
        "description": (
            "technical.structure + technical.instruments — matches legacy UI payloads; "
            "synced to engine via CompleteSongIntentRequest."
        ),
        "value": {
            "intent_schema_version": 1,
            "intent": {
                "emotional_intent": "intimate and powerful pop ballad",
                "core_desire": "emotional pop ballad",
                "technical": {
                    "genre": "pop",
                    "key": "C major",
                    "bpm": 120,
                    "structure": [
                        {"name": "intro", "bars": 4},
                        {"name": "verse", "bars": 8},
                        {"name": "chorus", "bars": 8},
                    ],
                    "instruments": [
                        {"instrument": "piano"},
                        {"instrument": "bass"},
                        {"instrument": "drums"},
                    ],
                },
            },
        },
    },
    "ttg_v1": {
        "summary": "TTG timeline + orchestration (PRD v1)",
        "description": (
            "Temporal Tension Graph root movement + role/patch thresholds; "
            "adapter flattens to engine structure/instruments."
        ),
        "value": {
            "intent_schema_version": 1,
            "intent": {
                "emotional_intent": "tension into release — club energy",
                "core_desire": "driving electronic track",
                "technical": {
                    "genre": "electronic",
                    "key": "D minor",
                    "bpm": 124,
                    "timeline": {
                        "type": "movement",
                        "id": "A",
                        "bars": 9,
                        "children": [
                            {
                                "bars": 8,
                                "section_role": "verse",
                                "harmonic_rhythm": "slow",
                                "boundary_event": "drum_fill_1bar",
                            }
                        ],
                    },
                    "orchestration": {
                        "roles": {
                            "sub_bass": {
                                "patch": "808_clean",
                                "active_threshold": 0.5,
                            },
                            "arp_synth": {
                                "patch": "juno_106",
                                "active_threshold": 0.2,
                            },
                        }
                    },
                    "energy_curve": {
                        "points": [
                            {"bar": 0.0, "value": 0.2},
                            {"bar": 16.0, "value": 0.95},
                        ]
                    },
                },
            },
        },
    },
}


class GenerateRequest(BaseModel):
    """
    HTTP /generate body (PRD §6–7).

    intent_schema_version: TTG-capable contract version; only 1 is supported today.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": GENERATE_REQUEST_OPENAPI_EXAMPLES["flat_v0"]["value"],
        }
    )

    intent_schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Intent document version for migrations (PRD §7).",
    )
    intent: EmotionalIntent
    output_format: Optional[str] = None


class InterrogateRequest(BaseModel):
    message: str = Field(..., max_length=4096)
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class LyricsRequest(BaseModel):
    lyrics: str = Field(..., max_length=32768)
    source: Optional[str] = "user"
