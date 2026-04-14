"""TTG v1 adapter and pipeline integration (PRD §8)."""

import pytest

from music_brain.api_schemas.generate_v1 import (
    EmotionalIntent,
    GenerateRequest,
    TechnicalIntent,
)
from music_brain.api_schemas.ttg_adapter import (
    orchestration_to_instruments,
    ttg_movement_to_structure_sections,
)
from music_brain.api_schemas.ttg_v1 import (
    OrchestrationRoleSpecV1,
    TTGMovementV1,
    TTGPhraseV1,
    TTGOrchestrationV1,
)
from music_brain.pipeline import IntentPipeline


def test_ttg_drum_fill_inserts_build_bar():
    movement = TTGMovementV1(
        id="A",
        bars=9,
        children=[
            TTGPhraseV1(bars=8, section_role="verse", boundary_event="drum_fill_1bar"),
        ],
    )
    sections = ttg_movement_to_structure_sections(movement)
    assert sections == [
        {"name": "verse", "bars": 8, "repetitions": 1},
        {"name": "build", "bars": 1, "repetitions": 1},
    ]


def test_ttg_bars_mismatch_raises():
    with pytest.raises(ValueError, match="does not match derived"):
        TTGMovementV1(
            id="A",
            bars=10,
            children=[TTGPhraseV1(bars=8, boundary_event="drum_fill_1bar")],
        )


def test_orchestration_sorted_roles():
    orch = TTGOrchestrationV1(
        roles={
            "z_role": OrchestrationRoleSpecV1(patch="z_patch", active_threshold=0.3),
            "a_role": OrchestrationRoleSpecV1(patch="a_patch", active_threshold=0.7),
        }
    )
    inst = orchestration_to_instruments(orch)
    assert inst[0]["instrument"] == "a_patch"
    assert inst[0]["techniques"] == ["role:a_role"]
    assert inst[1]["instrument"] == "z_patch"


def test_pipeline_normalize_prefers_ttg_over_flat_structure():
    tech = TechnicalIntent(
        key="D minor",
        bpm=100,
        genre="electronic",
        structure=[{"name": "chorus", "bars": 99, "repetitions": 1}],
        instruments=[{"instrument": "ignored", "techniques": []}],
        timeline=TTGMovementV1(
            id="A",
            bars=4,
            children=[TTGPhraseV1(bars=4, section_role="intro")],
        ),
        orchestration=TTGOrchestrationV1(
            roles={"bass": OrchestrationRoleSpecV1(patch="808_clean", active_threshold=0.5)}
        ),
    )
    req = GenerateRequest(
        intent=EmotionalIntent(
            emotional_intent="tension",
            core_desire="drop",
            technical=tech,
        )
    )
    pipeline = IntentPipeline()
    normalized = pipeline.normalize(req)
    assert normalized["structure"] == [
        {"name": "intro", "bars": 4, "repetitions": 1},
    ]
    assert normalized["instruments"] == [
        {"instrument": "808_clean", "techniques": ["role:bass"]},
    ]
    validated = pipeline.validate(normalized)
    assert validated.structure[0].name == "intro"
    assert validated.instruments[0].instrument == "808_clean"


def test_energy_curve_preserved_but_not_in_complete_request():
    tech = TechnicalIntent(
        key="C major",
        bpm=120,
        genre="pop",
        structure=[{"name": "verse", "bars": 8}],
        instruments=[{"instrument": "piano", "techniques": []}],
        energy_curve={
            "points": [{"bar": 0.0, "value": 0.2}, {"bar": 8.0, "value": 0.9}],
        },
    )
    req = GenerateRequest(intent=EmotionalIntent(emotional_intent="calm", technical=tech))
    pipeline = IntentPipeline()
    normalized = pipeline.normalize(req)
    assert "energy_curve" in normalized
    pipeline.validate(normalized)
