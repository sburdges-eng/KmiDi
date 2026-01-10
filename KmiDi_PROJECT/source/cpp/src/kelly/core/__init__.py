"""Core module for Kelly.

This package contains the core functionality for emotion processing,
intent analysis, and MIDI generation.
"""

from kelly.core.emotion_thesaurus import (
    EmotionThesaurus,
    EmotionNode,
    EmotionCategory,
)
from kelly.core.intent_processor import (
    IntentProcessor,
    Wound,
    RuleBreak,
    IntentPhase,
)
from kelly.core.midi_generator import (
    MidiGenerator,
    GrooveTemplate,
    Chord,
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
