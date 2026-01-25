"""Core module for Kelly.

This package contains the core functionality for emotion processing,
intent analysis, and MIDI generation.
"""

from kelly.core.emotion_thesaurus import (
    EmotionCategory,
    EmotionNode,
    EmotionThesaurus,
)
from kelly.core.intent_processor import (
    IntentPhase,
    IntentProcessor,
    RuleBreak,
    Wound,
)
from kelly.core.midi_generator import (
    Chord,
    GrooveTemplate,
    MidiGenerator,
)

__all__ = [
    "EmotionThesaurus",
    "EmotionNode",
    "EmotionCategory",
    "IntentProcessor",
    "Wound",
    "RuleBreak",
    "IntentPhase",
    "MidiGenerator",
    "GrooveTemplate",
    "Chord",
]
