"""
Feedback Interpreter - Maps natural language descriptions to parameters.

Provides preview and confirmation workflow for natural language editing.
"""

from typing import Dict, List, Optional, Any
from music_brain.editing.natural_language_processor import (
    NaturalLanguageProcessor,
    InterpretedFeedback,
    FeedbackInterpreter as BaseFeedbackInterpreter
)

# Re-export for convenience
__all__ = ["NaturalLanguageProcessor", "InterpretedFeedback", "FeedbackInterpreter"]

# Alias for backward compatibility
FeedbackInterpreter = BaseFeedbackInterpreter
