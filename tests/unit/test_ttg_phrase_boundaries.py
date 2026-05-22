"""TTG phrase-boundary extraction (Goal: Phrase-boundary prediction, Bar/beat structural awareness)."""

import pytest

from music_brain.api_schemas.ttg_adapter import extract_phrase_boundaries
from music_brain.api_schemas.ttg_v1 import TTGMovementV1, TTGPhraseV1


def test_no_boundaries_for_single_phrase():
    """A movement with one phrase has no internal boundaries."""
    movement = TTGMovementV1(
        id="A",
        bars=8,
        children=[TTGPhraseV1(bars=8, section_role="verse")],
    )
    assert extract_phrase_boundaries(movement) == []


def test_boundary_between_consecutive_phrases():
    """Two phrases → one boundary at the end of the first, at bar = phrase[0].bars."""
    movement = TTGMovementV1(
        id="A",
        bars=16,
        children=[
            TTGPhraseV1(bars=8, section_role="verse"),
            TTGPhraseV1(bars=8, section_role="chorus"),
        ],
    )
    assert extract_phrase_boundaries(movement) == [
        {
            "bar_offset": 8.0,
            "event": None,
            "from_section": "verse",
            "to_section": "chorus",
        },
    ]


def test_drum_fill_event_extends_offset_to_next_phrase_start():
    """drum_fill_1bar inserts a 1-bar fill, so the next phrase starts at offset+1."""
    movement = TTGMovementV1(
        id="A",
        bars=17,
        children=[
            TTGPhraseV1(bars=8, section_role="verse", boundary_event="drum_fill_1bar"),
            TTGPhraseV1(bars=8, section_role="chorus"),
        ],
    )
    boundaries = extract_phrase_boundaries(movement)
    assert boundaries == [
        {
            "bar_offset": 8.0,
            "event": "drum_fill_1bar",
            "from_section": "verse",
            "to_section": "chorus",
        },
    ]


def test_multiple_phrases_accumulate_bar_offsets():
    """Cumulative bar offset across multiple phrases (with and without fills)."""
    movement = TTGMovementV1(
        id="A",
        bars=25,  # 8 + 1 (fill) + 8 + 8
        children=[
            TTGPhraseV1(bars=8, section_role="verse", boundary_event="drum_fill_1bar"),
            TTGPhraseV1(bars=8, section_role="chorus"),
            TTGPhraseV1(bars=8, section_role="outro"),
        ],
    )
    boundaries = extract_phrase_boundaries(movement)
    assert len(boundaries) == 2
    assert boundaries[0]["bar_offset"] == pytest.approx(8.0)
    assert boundaries[0]["event"] == "drum_fill_1bar"
    assert boundaries[0]["from_section"] == "verse"
    assert boundaries[0]["to_section"] == "chorus"
    # Second boundary lands at 8 + 1 (fill) + 8 = 17
    assert boundaries[1]["bar_offset"] == pytest.approx(17.0)
    assert boundaries[1]["event"] is None
    assert boundaries[1]["from_section"] == "chorus"
    assert boundaries[1]["to_section"] == "outro"


def test_boundary_uses_inferred_section_when_role_absent():
    """When section_role is unset, infer_section_role provides the label."""
    movement = TTGMovementV1(
        id="A",
        bars=16,
        children=[
            TTGPhraseV1(bars=8),  # idx 0 of 2 → "intro"
            TTGPhraseV1(bars=8),  # idx 1 of 2 → "chorus"
        ],
    )
    boundaries = extract_phrase_boundaries(movement)
    assert boundaries == [
        {
            "bar_offset": 8.0,
            "event": None,
            "from_section": "intro",
            "to_section": "chorus",
        },
    ]


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_normalize_emits_phrase_boundaries():
    from music_brain.api_schemas.generate_v1 import (
        EmotionalIntent,
        GenerateRequest,
        TechnicalIntent,
    )
    from music_brain.pipeline import IntentPipeline

    tech = TechnicalIntent(
        key="C major",
        bpm=120,
        genre="pop",
        structure=[{"name": "verse", "bars": 8}],
        instruments=[{"instrument": "piano", "techniques": []}],
        timeline=TTGMovementV1(
            id="A",
            bars=17,
            children=[
                TTGPhraseV1(bars=8, section_role="verse", boundary_event="drum_fill_1bar"),
                TTGPhraseV1(bars=8, section_role="chorus"),
            ],
        ),
    )
    req = GenerateRequest(intent=EmotionalIntent(emotional_intent="hope", technical=tech))
    normalized = IntentPipeline().normalize(req)
    assert "phrase_boundaries" in normalized
    boundaries = normalized["phrase_boundaries"]
    assert len(boundaries) == 1
    assert boundaries[0]["bar_offset"] == pytest.approx(8.0)
    assert boundaries[0]["event"] == "drum_fill_1bar"
    assert boundaries[0]["from_section"] == "verse"
    assert boundaries[0]["to_section"] == "chorus"


def test_pipeline_normalize_emits_empty_boundaries_for_single_phrase():
    """Single-phrase timeline → phrase_boundaries omitted (empty list is noise)."""
    from music_brain.api_schemas.generate_v1 import (
        EmotionalIntent,
        GenerateRequest,
        TechnicalIntent,
    )
    from music_brain.pipeline import IntentPipeline

    tech = TechnicalIntent(
        key="C major",
        bpm=120,
        genre="pop",
        structure=[{"name": "verse", "bars": 8}],
        instruments=[{"instrument": "piano", "techniques": []}],
        timeline=TTGMovementV1(
            id="A",
            bars=8,
            children=[TTGPhraseV1(bars=8, section_role="verse")],
        ),
    )
    req = GenerateRequest(intent=EmotionalIntent(emotional_intent="hope", technical=tech))
    normalized = IntentPipeline().normalize(req)
    assert "phrase_boundaries" not in normalized


def test_pipeline_normalize_omits_boundaries_when_no_timeline():
    """Flat structure → no phrase boundaries (concept doesn't apply)."""
    from music_brain.api_schemas.generate_v1 import (
        EmotionalIntent,
        GenerateRequest,
        TechnicalIntent,
    )
    from music_brain.pipeline import IntentPipeline

    tech = TechnicalIntent(
        key="C major",
        bpm=120,
        genre="pop",
        structure=[{"name": "verse", "bars": 8}, {"name": "chorus", "bars": 8}],
        instruments=[{"instrument": "piano", "techniques": []}],
    )
    req = GenerateRequest(intent=EmotionalIntent(emotional_intent="hope", technical=tech))
    normalized = IntentPipeline().normalize(req)
    assert "phrase_boundaries" not in normalized
