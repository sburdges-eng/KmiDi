"""
Suggestion Bridge - Python interface for C++ to call SuggestionEngine.

Provides a simple function that C++ can call to get suggestions.
"""

from typing import Dict, List, Any, Optional
import json

from music_brain.intelligence.suggestion_engine import (
    SuggestionEngine,
    Suggestion,
    SuggestionType,
)
from music_brain.learning.user_preferences import UserPreferenceModel
from music_brain.intelligence.context_analyzer import ContextAnalyzer


# Global instances (singleton pattern)
_preference_model: Optional[UserPreferenceModel] = None
_suggestion_engine: Optional[SuggestionEngine] = None
_context_analyzer: Optional[ContextAnalyzer] = None


def initialize_suggestion_system(user_id: str = "default"):
    """
    Initialize the suggestion system with user preference model.

    Args:
        user_id: User identifier for preference tracking
    """
    global _preference_model, _suggestion_engine, _context_analyzer

    _preference_model = UserPreferenceModel(user_id=user_id)
    _context_analyzer = ContextAnalyzer()
    _suggestion_engine = SuggestionEngine(
        preference_model=_preference_model,
        context_analyzer=_context_analyzer
    )


def get_suggestions(
    current_state_json: str,
    max_suggestions: int = 5
) -> str:
    """
    Get suggestions based on current musical state.

    This function is designed to be called from C++ via Python bridge.

    Args:
        current_state_json: JSON string containing current state:
            {
                "parameters": {"valence": -0.5, "arousal": 0.4, ...},
                "emotion": "grief",
                "rule_breaks": ["HARMONY_AvoidTonicResolution"],
                ...
            }
        max_suggestions: Maximum number of suggestions to return

    Returns:
        JSON string containing list of suggestions:
        [
            {
                "suggestion_type": "parameter",
                "title": "Increase valence",
                "description": "...",
                "action": {"parameter": "valence", "target_value": 0.2},
                "confidence": 0.75,
                "explanation": "...",
                "source": "user_history"
            },
            ...
        ]
    """
    global _suggestion_engine, _preference_model

    # Initialize if not already done
    if _suggestion_engine is None:
        initialize_suggestion_system()

    try:
        # Parse current state
        current_state = json.loads(current_state_json)

        # Generate suggestions
        suggestions = _suggestion_engine.generate_suggestions(
            current_state,
            max_suggestions=max_suggestions
        )

        # Convert suggestions to JSON-serializable format
        suggestions_dict = []
        for suggestion in suggestions:
            suggestions_dict.append({
                "suggestion_type": suggestion.suggestion_type.value,
                "title": suggestion.title,
                "description": suggestion.description,
                "action": suggestion.action,
                "confidence": suggestion.confidence,
                "explanation": suggestion.explanation,
                "source": suggestion.source,
            })

        return json.dumps(suggestions_dict)

    except Exception as e:
        # Return empty list on error
        return json.dumps([])


def record_suggestion_shown(
    suggestion_id: str,
    suggestion_type: str,
    context_json: str = "{}"
):
    """
    Record that a suggestion was shown to the user.

    Args:
        suggestion_id: Unique identifier for the suggestion
        suggestion_type: Type of suggestion ("parameter", "emotion", etc.)
        context_json: JSON string with context information
    """
    global _preference_model

    if _preference_model is None:
        initialize_suggestion_system()

    try:
        context = json.loads(context_json) if context_json else {}
        _preference_model.record_suggestion_shown(
            suggestion_id=suggestion_id,
            suggestion_type=suggestion_type,
            context=context
        )
    except Exception as e:
        pass  # Silently fail - tracking is not critical


def record_suggestion_accepted(suggestion_id: str):
    """
    Record that user accepted (applied) a suggestion.

    Args:
        suggestion_id: Unique identifier for the suggestion
    """
    global _preference_model

    if _preference_model is None:
        initialize_suggestion_system()

    try:
        _preference_model.record_suggestion_accepted(suggestion_id)
    except Exception as e:
        pass  # Silently fail - tracking is not critical


def record_suggestion_dismissed(suggestion_id: str):
    """
    Record that user dismissed a suggestion.

    Args:
        suggestion_id: Unique identifier for the suggestion
    """
    global _preference_model

    if _preference_model is None:
        initialize_suggestion_system()

    try:
        _preference_model.record_suggestion_dismissed(suggestion_id)
    except Exception as e:
        pass  # Silently fail - tracking is not critical
