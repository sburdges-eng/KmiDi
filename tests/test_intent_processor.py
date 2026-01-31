"""
Unit tests for music_brain.session.intent_processor (process_intent, IntentProcessor).

Per PROJECT_ROADMAP_REIMPLEMENTATION.md Phase 3: key/mode and rule_break.
"""
import pytest


def test_process_intent_default_key_mode():
    """process_intent with minimal intent defaults key F, mode major (IntentProcessor._parse_intent)."""
    from music_brain.session.intent_schema import CompleteSongIntent
    from music_brain.session.intent_processor import process_intent

    intent = CompleteSongIntent.from_flat(technical_key="", technical_mode="")
    result = process_intent(intent)
    assert "harmony" in result
    assert result["harmony"].key == "F"
    assert result["harmony"].mode == "major"
    assert isinstance(result["harmony"].chords, list)
    assert len(result["harmony"].chords) >= 1


def test_process_intent_harmony_modal_interchange():
    """process_intent with HARMONY_ModalInterchange returns progression with rule_broken set."""
    from music_brain.session.intent_schema import CompleteSongIntent
    from music_brain.session.intent_processor import process_intent

    intent = CompleteSongIntent.from_flat(
        technical_key="F",
        technical_mode="major",
        technical_rule_to_break="HARMONY_ModalInterchange",
    )
    result = process_intent(intent)
    assert result["harmony"].rule_broken == "HARMONY_ModalInterchange"
    assert result["harmony"].key == "F"
    assert result["harmony"].mode == "major"
    assert len(result["harmony"].chords) >= 1


def test_process_intent_returns_groove_arrangement_production():
    """process_intent returns harmony, groove, arrangement, production, intent_summary."""
    from music_brain.session.intent_schema import CompleteSongIntent
    from music_brain.session.intent_processor import process_intent

    intent = CompleteSongIntent.from_flat(technical_key="C", technical_mode="major")
    result = process_intent(intent)
    assert "harmony" in result
    assert "groove" in result
    assert "arrangement" in result
    assert "production" in result
    assert "intent_summary" in result
    assert "mood" in result["intent_summary"]
    assert "rule_broken" in result["intent_summary"]
