"""Tests for emotion thesaurus."""
import pytest
from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionCategory


def test_emotion_thesaurus_initialization():
    """Test thesaurus initializes with emotions."""
    thesaurus = EmotionThesaurus()
    assert len(thesaurus.nodes) > 0


def test_get_emotion_by_id():
    """Test retrieving emotion by ID."""
    thesaurus = EmotionThesaurus()
    emotion = thesaurus.get_emotion(0)
    assert emotion is not None
    assert emotion.name == "euphoria"
    assert emotion.category == EmotionCategory.JOY


def test_find_emotion_by_name():
    """Test finding emotion by name."""
    thesaurus = EmotionThesaurus()
    emotion = thesaurus.find_emotion_by_name("grief")
    assert emotion is not None
    assert emotion.id == 2
    assert emotion.category == EmotionCategory.SADNESS


def test_get_nearby_emotions():
    """Test finding nearby emotions in emotional space."""
    thesaurus = EmotionThesaurus()
    nearby = thesaurus.get_nearby_emotions(0, threshold=0.5)
    assert isinstance(nearby, list)


def test_emotion_musical_attributes():
    """Test emotions have musical attributes."""
    thesaurus = EmotionThesaurus()
    emotion = thesaurus.get_emotion(0)
    assert "tempo_modifier" in emotion.musical_attributes
    assert "mode" in emotion.musical_attributes
    assert "dynamics" in emotion.musical_attributes


def test_invalid_emotion_id():
    """Test getting invalid emotion ID returns None."""
    thesaurus = EmotionThesaurus()
    emotion = thesaurus.get_emotion(9999)
    assert emotion is None
