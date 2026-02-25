"""Unit tests for core modules."""

import pytest
from music_brain.session.intent_schema import CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints
from music_brain.session.intent_processor import IntentProcessor


class TestIntentSchema:
    """Test intent schema classes."""
    
    def test_complete_song_intent_creation(self, sample_intent):
        """Test creating a CompleteSongIntent."""
        assert sample_intent is not None
        assert sample_intent.song_root is not None
        assert sample_intent.song_intent is not None
        assert sample_intent.technical_constraints is not None
    
    def test_song_root_fields(self):
        """Test SongRoot fields."""
        root = SongRoot(
            core_event="Test event",
            core_longing="Test longing"
        )
        assert root.core_event == "Test event"
        assert root.core_longing == "Test longing"
    
    def test_song_intent_fields(self):
        """Test SongIntent fields."""
        intent = SongIntent(mood_primary="joyful")
        assert hasattr(intent, 'mood_primary')


class TestIntentProcessor:
    """Test IntentProcessor class."""
    
    def test_processor_creation(self, sample_intent):
        """Test creating an IntentProcessor."""
        processor = IntentProcessor(sample_intent)
        assert processor.intent == sample_intent
        assert hasattr(processor, 'key')
        assert hasattr(processor, 'mode')
