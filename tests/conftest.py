"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402


@pytest.fixture
def project_root_path():
    """Return the project root directory."""
    return project_root


@pytest.fixture
def sample_intent():
    """Create a sample CompleteSongIntent for testing."""
    from music_brain.session.intent_schema import (
        CompleteSongIntent,
        SongRoot,
        SongIntent,
        TechnicalConstraints,
    )

    return CompleteSongIntent(
        song_root=SongRoot(core_event="Test event"),
        song_intent=SongIntent(mood_primary="joyful"),
        technical_constraints=TechnicalConstraints(technical_key="C"),
    )


@pytest.fixture
def sample_emotion_thesaurus():
    """Create a sample EmotionThesaurus instance."""
    from music_brain.kelly_companion.core.emotion_thesaurus import EmotionThesaurus

    return EmotionThesaurus()
