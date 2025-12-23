"""Tests for intent processor."""
import pytest
from kelly.core.intent_processor import IntentProcessor, Wound, IntentPhase


def test_wound_creation():
    """Test creating a wound object."""
    wound = Wound(
        description="feeling of loss",
        intensity=0.8,
        source="user_input"
    )
    assert wound.description == "feeling of loss"
    assert wound.intensity == 0.8


def test_process_wound():
    """Test processing a wound returns an emotion."""
    processor = IntentProcessor()
    wound = Wound(description="grief and loss", intensity=0.9, source="user")
    emotion = processor.process_wound(wound)
    assert emotion is not None
    assert emotion.name == "grief"


def test_emotion_to_rule_breaks():
    """Test converting emotion to rule breaks."""
    processor = IntentProcessor()
    emotion = processor.thesaurus.find_emotion_by_name("rage")
    rule_breaks = processor.emotion_to_rule_breaks(emotion)
    assert len(rule_breaks) > 0
    assert any(rb.rule_type == "dynamics" for rb in rule_breaks)


def test_process_intent_complete():
    """Test complete intent processing pipeline."""
    processor = IntentProcessor()
    wound = Wound(description="feeling anxious", intensity=0.7, source="user")
    result = processor.process_intent(wound)
    
    assert "wound" in result
    assert "emotion" in result
    assert "rule_breaks" in result
    assert "musical_params" in result


def test_wound_history():
    """Test wound history is tracked."""
    processor = IntentProcessor()
    wound1 = Wound(description="loss", intensity=0.8, source="user")
    wound2 = Wound(description="anger", intensity=0.6, source="user")
    
    processor.process_wound(wound1)
    processor.process_wound(wound2)
    
    assert len(processor.wound_history) == 2
