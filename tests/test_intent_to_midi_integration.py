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
