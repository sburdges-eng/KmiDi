"""
Feedback Interpreter - Maps natural language descriptions to parameters.

Provides preview and confirmation workflow for natural language editing.
"""

from music_brain.editing.natural_language_processor import (
    FeedbackInterpreter as BaseFeedbackInterpreter,
)
from music_brain.editing.natural_language_processor import (
    InterpretedFeedback,
    NaturalLanguageProcessor,
)

# Re-export for convenience
__all__ = ["NaturalLanguageProcessor", "InterpretedFeedback", "FeedbackInterpreter"]

# Alias for backward compatibility
FeedbackInterpreter = BaseFeedbackInterpreter
