"""
Editing module for music brain.

Provides MIDI and sheet music editing capabilities.
"""

from .feedback_interpreter import (
    FeedbackInterpreter,
)
from .midi_editor import (
    EditOperation,
    MidiEditCommand,
    MidiEditor,
)
from .natural_language_processor import (
    FeedbackType,
    Intent,
    InterpretedFeedback,
    NaturalLanguageProcessor,
)

__all__ = [
    # MIDI editing
    "MidiEditor",
    "EditOperation",
    "MidiEditCommand",
    # Natural language processing
    "NaturalLanguageProcessor",
    "InterpretedFeedback",
    "FeedbackType",
    "Intent",
    "FeedbackInterpreter",
]
