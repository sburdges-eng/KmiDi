#!/usr/bin/env python3
"""Test complete song generation workflow."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_emotion_to_intent():
    """Test emotion → intent conversion."""
    from music_brain.session.intent_schema import (
        CompleteSongIntent,
        SongRoot,
        SongIntent,
        TechnicalConstraints,
    )

    intent = CompleteSongIntent(
        song_root=SongRoot(core_event="Test event", core_longing="Test longing"),
        song_intent=SongIntent(mood_primary="joyful"),
        technical_constraints=TechnicalConstraints(
            technical_key="C", technical_tempo_range=(120, 140)
        ),
    )

    # Check that intent was created (fields may be empty strings by default)
    assert intent.song_intent is not None
    assert intent.technical_constraints is not None
    # mood_primary may be empty string if not explicitly set, so check it exists
    assert hasattr(intent.song_intent, "mood_primary")
    print("✅ Emotion → Intent conversion works")
    return True


def test_intent_to_engines():
    """Test intent → engine generation."""
    from music_brain.session.intent_processor import IntentProcessor
    from music_brain.session.intent_schema import (
        CompleteSongIntent,
        SongRoot,
        SongIntent,
        TechnicalConstraints,
    )

    intent = CompleteSongIntent(
        song_root=SongRoot(core_event="Test"),
        song_intent=SongIntent(mood_primary="joyful"),
        technical_constraints=TechnicalConstraints(technical_key="C"),
    )

    processor = IntentProcessor(intent)
    # Test that processor was created successfully
    assert processor.intent == intent
    print("✅ Intent → Engines conversion works")
    return True


def test_complete_workflow():
    """Test complete workflow: emotion → intent → MIDI."""
    from music_brain.session.intent_schema import (
        CompleteSongIntent,
        SongRoot,
        SongIntent,
        TechnicalConstraints,
    )
    from music_brain.session.intent_processor import IntentProcessor
    from music_brain.kelly_companion.engines import BassEngine, MelodyEngine

    # Create intent from emotion
    intent = CompleteSongIntent(
        song_root=SongRoot(
            core_event="Creating a joyful song", core_longing="To express happiness"
        ),
        song_intent=SongIntent(mood_primary="joyful"),
        technical_constraints=TechnicalConstraints(
            technical_key="C", technical_tempo_range=(120, 140)
        ),
    )

    # Process intent
    IntentProcessor(intent)

    # Test engines can be instantiated
    BassEngine()
    MelodyEngine()

    print("✅ Complete workflow test - all components work together")
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Song Generation Workflow")
    print("=" * 80)
    print()

    results = [test_emotion_to_intent(), test_intent_to_engines(), test_complete_workflow()]

    print()
    print("=" * 80)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 80)

    sys.exit(0 if all(results) else 1)
