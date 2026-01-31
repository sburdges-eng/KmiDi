"""
Integration test: intent → process_intent → MIDIGenerationPipeline.generate_midi (no LLM).

Per PROJECT_ROADMAP_REIMPLEMENTATION.md Phase 1 & 3: critical path test.
"""
import tempfile
from pathlib import Path

import pytest


def test_intent_from_flat_to_midi_completed():
    """CompleteSongIntent.from_flat(...) → process_intent → MIDIGenerationPipeline.generate_midi → file exists, status completed."""
    from music_brain.session.intent_schema import CompleteSongIntent
    from music_brain.tier1.midi_pipeline_wrapper import MIDIGenerationPipeline

    intent = CompleteSongIntent.from_flat(
        technical_key="C",
        technical_mode="major",
        technical_rule_to_break="HARMONY_ModalInterchange",
        explanation="integration test",
    )
    pipeline = MIDIGenerationPipeline()
    with tempfile.TemporaryDirectory() as tmp:
        out = pipeline.generate_midi(intent, output_dir=tmp)
        assert out.get("status") == "completed"
        assert "midi_path" in out
        assert Path(out["midi_path"]).exists()
    assert isinstance(out.get("chords"), list)
    assert len(out.get("chords", [])) >= 1


def test_intent_to_midi_returns_expected_keys():
    """Pipeline return dict has status, midi_path, chords, rule_broken or details."""
    from music_brain.session.intent_schema import CompleteSongIntent
    from music_brain.tier1.midi_pipeline_wrapper import MIDIGenerationPipeline

    intent = CompleteSongIntent.from_flat(technical_key="F", technical_mode="major")
    pipeline = MIDIGenerationPipeline()
    with tempfile.TemporaryDirectory() as tmp:
        out = pipeline.generate_midi(intent, output_dir=tmp)
    assert "status" in out
    assert "midi_path" in out
    assert "chords" in out
    assert out["status"] in ("completed", "error")


def test_midi_pipeline_return_includes_integration_summaries_when_completed():
    """When status is completed, midi_plan-style return includes arrangement/melody/texture/temporal summary keys."""
    from music_brain.session.intent_schema import CompleteSongIntent
    from music_brain.tier1.midi_pipeline_wrapper import MIDIGenerationPipeline

    intent = CompleteSongIntent.from_flat(technical_key="F", technical_mode="major")
    pipeline = MIDIGenerationPipeline()
    with tempfile.TemporaryDirectory() as tmp:
        out = pipeline.generate_midi(intent, output_dir=tmp)
    if out.get("status") == "completed":
        assert "groove_tempo" in out
        assert "arrangement_summary" in out
        assert "melody_contour" in out
        assert "texture_density" in out
        assert "temporal_pacing" in out


def test_user_text_to_midi_via_llm_engine():
    """User text → LLMReasoningEngine.parse_user_intent (rule-based) → MIDIGenerationPipeline.generate_midi → completed."""
    from music_brain.tier1.midi_pipeline_wrapper import MIDIGenerationPipeline
    from mcp_workstation.llm_reasoning_engine import LLMReasoningEngine

    engine = LLMReasoningEngine(model_path="")
    intent = engine.parse_user_intent("Something nostalgic in F major with borrowed chords.")
    assert hasattr(intent, "song_root") and hasattr(intent, "technical_constraints")
    assert intent.technical_constraints.technical_key == "F"
    pipeline = MIDIGenerationPipeline()
    with tempfile.TemporaryDirectory() as tmp:
        out = pipeline.generate_midi(intent, output_dir=tmp)
        assert out.get("status") == "completed"
        assert "midi_path" in out
        assert Path(out["midi_path"]).exists()


def test_api_process_song_intent_includes_melody_texture_temporal():
    """Contract: process_intent output serialized like API process_song_intent includes melody, texture, temporal."""
    from music_brain.session.intent_schema import CompleteSongIntent
    from music_brain.session.intent_processor import process_intent

    intent = CompleteSongIntent.from_flat(technical_key="C", technical_mode="major")
    result = process_intent(intent)
    # Same serialization as api.process_song_intent (melody/texture/temporal blocks)
    out = {"harmony": True, "groove": True}
    if "melody" in result:
        m = result["melody"]
        out["melody"] = {
            "contour": m.contour,
            "interval_character": m.interval_character,
            "phrase_structure": m.phrase_structure,
            "resolution_behavior": m.resolution_behavior,
            "rhythmic_character": m.rhythmic_character,
            "range_notes": m.range_notes,
            "motif_ideas": getattr(m, "motif_ideas", []) or [],
            "rule_broken": getattr(m, "rule_broken", "") or "",
            "rule_effect": getattr(m, "rule_effect", "") or "",
        }
    if "texture" in result:
        t = result["texture"]
        out["texture"] = {
            "density_level": t.density_level,
            "frequency_balance": t.frequency_balance,
            "space_character": getattr(t, "space_character", "") or "",
            "rule_broken": getattr(t, "rule_broken", "") or "",
            "rule_effect": getattr(t, "rule_effect", "") or "",
        }
    if "temporal" in result:
        tm = result["temporal"]
        out["temporal"] = {
            "pacing": tm.pacing,
            "pause_strategy": getattr(tm, "pause_strategy", "") or "",
            "transition_style": getattr(tm, "transition_style", "") or "",
            "time_feel": getattr(tm, "time_feel", "") or "",
            "rule_broken": getattr(tm, "rule_broken", "") or "",
            "rule_effect": getattr(tm, "rule_effect", "") or "",
        }
    assert "melody" in out and "texture" in out and "temporal" in out
    assert "contour" in out["melody"]
    assert "density_level" in out["texture"]
    assert "pacing" in out["temporal"]
