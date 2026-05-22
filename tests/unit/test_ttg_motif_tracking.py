"""TTG motif recurrence tracking (Goal: Motif recurrence tracking, Semantic motif persistence)."""

import pytest

from music_brain.api_schemas.ttg_adapter import (
    extract_motif_inventory,
    summarize_motif_recurrence,
)
from music_brain.api_schemas.ttg_v1 import TTGMovementV1, TTGPhraseV1


# ---------------------------------------------------------------------------
# extract_motif_inventory
# ---------------------------------------------------------------------------


def test_extract_inventory_empty_when_no_motifs():
    movement = TTGMovementV1(
        id="A",
        bars=8,
        children=[TTGPhraseV1(bars=8, section_role="verse")],
    )
    assert extract_motif_inventory(movement) == {}


def test_extract_inventory_singleton_motif():
    movement = TTGMovementV1(
        id="A",
        bars=8,
        children=[TTGPhraseV1(bars=8, section_role="verse", motifs=["theme_a"])],
    )
    inv = extract_motif_inventory(movement)
    assert "theme_a" in inv
    assert inv["theme_a"] == [
        {"phrase_index": 0, "section": "verse", "bars": 8},
    ]


def test_extract_inventory_motif_recurs_across_phrases():
    movement = TTGMovementV1(
        id="A",
        bars=24,
        children=[
            TTGPhraseV1(bars=8, section_role="verse", motifs=["theme_a", "ostinato"]),
            TTGPhraseV1(bars=8, section_role="chorus", motifs=["theme_a"]),
            TTGPhraseV1(bars=8, section_role="verse", motifs=["theme_a", "ostinato"]),
        ],
    )
    inv = extract_motif_inventory(movement)
    assert inv["theme_a"] == [
        {"phrase_index": 0, "section": "verse", "bars": 8},
        {"phrase_index": 1, "section": "chorus", "bars": 8},
        {"phrase_index": 2, "section": "verse", "bars": 8},
    ]
    assert inv["ostinato"] == [
        {"phrase_index": 0, "section": "verse", "bars": 8},
        {"phrase_index": 2, "section": "verse", "bars": 8},
    ]


def test_extract_inventory_uses_inferred_section_when_role_absent():
    """When phrase.section_role is None, the inventory uses the same inferred
    label as ttg_movement_to_structure_sections (infer_section_role)."""
    movement = TTGMovementV1(
        id="A",
        bars=24,
        children=[
            TTGPhraseV1(bars=8, motifs=["a"]),  # idx 0 of 3 → "intro"
            TTGPhraseV1(bars=8, motifs=["a"]),  # idx 1 of 3 → "verse"
            TTGPhraseV1(bars=8, motifs=["a"]),  # idx 2 of 3 → "outro"
        ],
    )
    inv = extract_motif_inventory(movement)
    sections = [occ["section"] for occ in inv["a"]]
    assert sections == ["intro", "verse", "outro"]


# ---------------------------------------------------------------------------
# summarize_motif_recurrence
# ---------------------------------------------------------------------------


def test_recurrence_summary_marks_singleton_motifs():
    """A motif that appears in only one phrase is flagged is_recurring=False."""
    movement = TTGMovementV1(
        id="A",
        bars=16,
        children=[
            TTGPhraseV1(bars=8, section_role="verse", motifs=["a", "b"]),
            TTGPhraseV1(bars=8, section_role="chorus", motifs=["a"]),
        ],
    )
    summary = summarize_motif_recurrence(movement)
    assert summary["a"]["occurrences"] == 2
    assert summary["a"]["is_recurring"] is True
    assert summary["a"]["first_section"] == "verse"
    assert summary["a"]["last_section"] == "chorus"
    assert summary["b"]["occurrences"] == 1
    assert summary["b"]["is_recurring"] is False
    assert summary["b"]["first_section"] == "verse"
    assert summary["b"]["last_section"] == "verse"


def test_recurrence_summary_empty_when_no_motifs():
    movement = TTGMovementV1(
        id="A",
        bars=8,
        children=[TTGPhraseV1(bars=8, section_role="verse")],
    )
    assert summarize_motif_recurrence(movement) == {}


def test_recurrence_summary_recurrence_factor():
    """recurrence_factor = occurrences / total_phrases (0..1)."""
    movement = TTGMovementV1(
        id="A",
        bars=32,
        children=[
            TTGPhraseV1(bars=8, section_role="verse", motifs=["a"]),
            TTGPhraseV1(bars=8, section_role="chorus", motifs=["a"]),
            TTGPhraseV1(bars=8, section_role="verse", motifs=["a"]),
            TTGPhraseV1(bars=8, section_role="outro"),  # no motif
        ],
    )
    summary = summarize_motif_recurrence(movement)
    assert summary["a"]["occurrences"] == 3
    assert summary["a"]["recurrence_factor"] == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# Pipeline integration: normalize() emits motif_inventory when timeline+motifs present
# ---------------------------------------------------------------------------


def test_pipeline_normalize_emits_motif_inventory():
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
            bars=16,
            children=[
                TTGPhraseV1(bars=8, section_role="verse", motifs=["theme_a"]),
                TTGPhraseV1(bars=8, section_role="chorus", motifs=["theme_a", "fill_b"]),
            ],
        ),
    )
    req = GenerateRequest(intent=EmotionalIntent(emotional_intent="hope", technical=tech))
    normalized = IntentPipeline().normalize(req)
    assert "motif_inventory" in normalized
    inv = normalized["motif_inventory"]
    assert sorted(inv.keys()) == ["fill_b", "theme_a"]
    assert len(inv["theme_a"]) == 2
    assert inv["theme_a"][0]["section"] == "verse"
    assert inv["theme_a"][1]["section"] == "chorus"


def test_pipeline_normalize_omits_motif_inventory_when_no_motifs():
    """No motifs anywhere → no `motif_inventory` key in normalized."""
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
    assert "motif_inventory" not in normalized


def test_pipeline_normalize_omits_motif_inventory_when_no_timeline():
    """No TTG timeline → motif_inventory absent (flat structure has no motif concept)."""
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
    )
    req = GenerateRequest(intent=EmotionalIntent(emotional_intent="hope", technical=tech))
    normalized = IntentPipeline().normalize(req)
    assert "motif_inventory" not in normalized
