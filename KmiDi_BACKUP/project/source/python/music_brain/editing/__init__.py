"""
Editing module for music brain.

Provides MIDI and sheet music editing capabilities.
"""

from .midi_editor import (
    MidiEditor,
    EditOperation,
    MidiEditCommand,
)

from .natural_language_processor import (
    NaturalLanguageProcessor,
    InterpretedFeedback,
    FeedbackType,
    Intent,
)

from .feedback_interpreter import (
    FeedbackInterpreter,
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
