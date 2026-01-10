"""
Engine Bridge - Python interface for C++ engine-level intelligence.

Provides functions that C++ engines can call to get intelligent suggestions
for melody, bass, drum, pad, string, and other engine types.

This module integrates with SuggestionEngine to provide context-aware
suggestions for individual engines.
"""

from typing import Dict, List, Any, Optional
import json
import threading

from music_brain.intelligence.suggestion_engine import (
    SuggestionEngine,
    Suggestion,
    SuggestionType,
)
from music_brain.intelligence.context_analyzer import ContextAnalyzer
from music_brain.learning.user_preferences import UserPreferenceModel


# Global instances (singleton pattern) with thread safety
_preference_model: Optional[UserPreferenceModel] = None
_suggestion_engine: Optional[SuggestionEngine] = None
_context_analyzer: Optional[ContextAnalyzer] = None
_engine_lock = threading.Lock()


def initialize_engine_system(user_id: str = "default"):
    """
    Initialize the engine intelligence system.

    Args:
        user_id: User identifier for preference tracking
    """
    global _preference_model, _suggestion_engine, _context_analyzer

    with _engine_lock:
        _preference_model = UserPreferenceModel(user_id=user_id)
        _context_analyzer = ContextAnalyzer()
        _suggestion_engine = SuggestionEngine(
            preference_model=_preference_model,
            context_analyzer=_context_analyzer
        )


def get_engine_suggestions(
    engine_type: str,
    current_state_json: str
) -> str:
    """
    Get suggestions for a specific engine type.

    This function is designed to be called from C++ via Python bridge.

    Args:
        engine_type: Engine type ("melody", "bass", "drum", "pad", "string", etc.)
        current_state_json: JSON string with current state:
            {
                "emotion": "grief",
                "key": "C",
                "mode": "minor",
                "chords": ["Am", "Dm", "F", "C"],
                "parameters": {"complexity": 0.4, "density": 0.3, ...},
                "context": {"emotion_category": "negative_low_energy", ...}
            }

    Returns:
        JSON string with engine-specific suggestions:
        {
            "contour": "descending",
            "density": "sparse",
            "velocity_range": [40, 75],
            "register_range": [55, 75],
            "parameter_adjustments": {"complexity": 0.3, "density": 0.2},
            "confidence": 0.85
        }
    """
    global _suggestion_engine, _preference_model

    # Initialize if not already done (with double-check locking)
    if _suggestion_engine is None:
        with _engine_lock:
            if _suggestion_engine is None:
                initialize_engine_system()

    try:
        # Parse current state
        current_state = json.loads(current_state_json)

        # Get context analysis
        context = None
        if _context_analyzer:
            context = _context_analyzer.analyze(current_state)

        # Generate engine-specific suggestions
        suggestions = _generate_engine_specific_suggestions(
            engine_type,
            current_state,
            context
        )

        return json.dumps(suggestions)

    except Exception as e:
        # Return default suggestions on error
        return json.dumps(_get_default_suggestions(engine_type))


def get_batch_engine_suggestions(
    engine_types: List[str],
    current_state_json: str
) -> str:
    """
    Get suggestions for multiple engines at once.

    Args:
        engine_types: List of engine types
        current_state_json: Current state JSON

    Returns:
        JSON string with suggestions for all engines:
        {
            "melody": {...},
            "bass": {...},
            "drum": {...}
        }
    """
    try:
        current_state = json.loads(current_state_json)
        batch_suggestions = {}

        for engine_type in engine_types:
            suggestions = _generate_engine_specific_suggestions(
                engine_type,
                current_state,
                None  # Context will be analyzed per engine if needed
            )
            batch_suggestions[engine_type] = suggestions

        return json.dumps(batch_suggestions)

    except Exception as e:
        return json.dumps({})


def record_suggestion_applied(
    engine_type: str,
    suggestion_json: str,
    result_json: str = "{}"
):
    """
    Record that engine suggestions were applied.

    Args:
        engine_type: Engine type
        suggestion_json: The suggestion that was applied
        result_json: Result of applying the suggestion (for learning)
    """
    global _preference_model

    if _preference_model is None:
        initialize_engine_system()

    try:
        suggestion = json.loads(suggestion_json)
        result = json.loads(result_json) if result_json else {}

        # Record in preference model for learning
        # This helps improve future suggestions
        _preference_model.record_engine_suggestion_applied(
            engine_type=engine_type,
            suggestion=suggestion,
            result=result
        )
    except Exception as e:
        pass  # Silently fail - tracking is not critical


def _generate_engine_specific_suggestions(
    engine_type: str,
    current_state: Dict[str, Any],
    context: Optional[Any]
) -> Dict[str, Any]:
    """
    Generate engine-specific suggestions based on state and context.

    Args:
        engine_type: Engine type
        current_state: Current state dictionary
        context: Context analysis (optional)

    Returns:
        Dictionary with engine-specific suggestions
    """
    emotion = current_state.get("emotion", "neutral").lower()
    parameters = current_state.get("parameters", {})
    chords = current_state.get("chords", [])

    # Base suggestions from emotion and context
    suggestions = {
        "confidence": 0.7,
        "parameter_adjustments": {},
    }

    # Engine-specific logic
    if engine_type == "melody":
        suggestions.update(_get_melody_suggestions(emotion, parameters, context))
    elif engine_type == "bass":
        suggestions.update(_get_bass_suggestions(emotion, parameters, chords, context))
    elif engine_type == "drum" or engine_type == "drum_groove":
        suggestions.update(_get_drum_suggestions(emotion, parameters, context))
    elif engine_type == "pad":
        suggestions.update(_get_pad_suggestions(emotion, parameters, context))
    elif engine_type == "string":
        suggestions.update(_get_string_suggestions(emotion, parameters, context))
    else:
        # Generic suggestions for unknown engine types
        suggestions.update(_get_generic_suggestions(emotion, parameters, context))

    return suggestions


def _get_melody_suggestions(
    emotion: str,
    parameters: Dict[str, float],
    context: Optional[Any]
) -> Dict[str, Any]:
    """Get melody-specific suggestions."""
    suggestions = {}

    # Emotion-based contour suggestions
    emotion_contours = {
        "grief": "descending",
        "sadness": "descending",
        "longing": "arch",
        "hope": "ascending",
        "joy": "ascending",
        "anger": "jagged",
        "anxiety": "wave",
    }
    suggestions["contour"] = emotion_contours.get(emotion, "arch")

    # Density based on emotion and complexity
    complexity = parameters.get("complexity", 0.5)
    if emotion in ["grief", "sadness", "longing"]:
        suggestions["density"] = "sparse"
    elif emotion in ["anger", "anxiety"]:
        suggestions["density"] = "dense"
    else:
        suggestions["density"] = "moderate" if complexity > 0.5 else "sparse"

    # Velocity range
    if emotion in ["grief", "sadness"]:
        suggestions["velocity_range"] = [40, 75]
    elif emotion in ["anger", "joy"]:
        suggestions["velocity_range"] = [80, 120]
    else:
        suggestions["velocity_range"] = [60, 100]

    # Register range
    if emotion in ["grief", "vulnerability"]:
        suggestions["register_range"] = [55, 75]  # Lower register
    elif emotion in ["hope", "joy"]:
        suggestions["register_range"] = [70, 90]  # Higher register
    else:
        suggestions["register_range"] = [60, 85]

    # Parameter adjustments
    if emotion in ["grief", "sadness"]:
        suggestions["parameter_adjustments"]["complexity"] = 0.3
        suggestions["parameter_adjustments"]["density"] = 0.2
    elif emotion in ["anger", "anxiety"]:
        suggestions["parameter_adjustments"]["complexity"] = 0.7
        suggestions["parameter_adjustments"]["density"] = 0.8

    return suggestions


def _get_bass_suggestions(
    emotion: str,
    parameters: Dict[str, float],
    chords: List[str],
    context: Optional[Any]
) -> Dict[str, Any]:
    """Get bass-specific suggestions."""
    suggestions = {}

    # Bass pattern style
    if emotion in ["grief", "sadness"]:
        suggestions["pattern_style"] = "sparse_root_notes"
    elif emotion in ["anger", "defiance"]:
        suggestions["pattern_style"] = "aggressive_rhythmic"
    else:
        suggestions["pattern_style"] = "walking_bass"

    # Root note emphasis
    suggestions["root_note_emphasis"] = 0.8

    # Velocity
    if emotion in ["grief", "sadness"]:
        suggestions["velocity_range"] = [50, 80]
    else:
        suggestions["velocity_range"] = [70, 110]

    return suggestions


def _get_drum_suggestions(
    emotion: str,
    parameters: Dict[str, float],
    context: Optional[Any]
) -> Dict[str, Any]:
    """Get drum/rhythm-specific suggestions."""
    suggestions = {}

    # Groove style
    if emotion in ["grief", "sadness"]:
        suggestions["groove_style"] = "sparse_halftime"
    elif emotion in ["anger", "defiance"]:
        suggestions["groove_style"] = "aggressive_four_on_floor"
    elif emotion in ["hope", "joy"]:
        suggestions["groove_style"] = "upbeat_shuffle"
    else:
        suggestions["groove_style"] = "straight"

    # Humanization
    if emotion in ["grief", "vulnerability"]:
        suggestions["humanization"] = 0.6  # More human feel
    else:
        suggestions["humanization"] = 0.4

    # Velocity range
    suggestions["velocity_range"] = [60, 100]

    return suggestions


def _get_pad_suggestions(
    emotion: str,
    parameters: Dict[str, float],
    context: Optional[Any]
) -> Dict[str, Any]:
    """Get pad-specific suggestions."""
    suggestions = {}

    # Pad texture
    if emotion in ["grief", "sadness"]:
        suggestions["texture"] = "dark_warm"
    elif emotion in ["hope", "joy"]:
        suggestions["texture"] = "bright_airy"
    else:
        suggestions["texture"] = "neutral"

    # Density
    suggestions["density"] = "moderate"

    # Velocity
    suggestions["velocity_range"] = [40, 70]  # Pads are typically softer

    return suggestions


def _get_string_suggestions(
    emotion: str,
    parameters: Dict[str, float],
    context: Optional[Any]
) -> Dict[str, Any]:
    """Get string-specific suggestions."""
    suggestions = {}

    # Articulation
    if emotion in ["grief", "sadness"]:
        suggestions["articulation"] = "legato"
    elif emotion in ["anger", "defiance"]:
        suggestions["articulation"] = "staccato"
    else:
        suggestions["articulation"] = "tenuto"

    # Velocity
    suggestions["velocity_range"] = [50, 90]

    return suggestions


def _get_generic_suggestions(
    emotion: str,
    parameters: Dict[str, float],
    context: Optional[Any]
) -> Dict[str, Any]:
    """Get generic suggestions for unknown engine types."""
    return {
        "confidence": 0.5,
        "parameter_adjustments": {},
    }


def _get_default_suggestions(engine_type: str) -> Dict[str, Any]:
    """Get default suggestions when analysis fails."""
    return {
        "confidence": 0.5,
        "parameter_adjustments": {},
    }
