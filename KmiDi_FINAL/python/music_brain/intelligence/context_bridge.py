"""
Context Bridge - Python interface for C++ context analysis.

Provides functions that C++ can call to analyze musical context and get
context-aware parameter adjustments.
"""

from typing import Dict, List, Any, Optional
import json
import threading

from music_brain.intelligence.context_analyzer import ContextAnalyzer, MusicalContext


# Global instance (singleton pattern) with thread safety
_context_analyzer: Optional[ContextAnalyzer] = None
_context_lock = threading.Lock()


def initialize_context_system():
    """Initialize the context analysis system."""
    global _context_analyzer
    with _context_lock:
        if _context_analyzer is None:
            _context_analyzer = ContextAnalyzer()


def analyze_context(state_json: str) -> str:
    """
    Analyze current musical context.

    This function is designed to be called from C++ via Python bridge.

    Args:
        state_json: JSON string with current state:
            {
                "emotion": "grief",
                "parameters": {"valence": -0.5, "arousal": 0.4, ...},
                "chords": ["Am", "Dm", "F", "C"],
                "current_section": "verse",
                ...
            }

    Returns:
        JSON string with context analysis:
        {
            "emotion_category": "negative_low_energy",
            "complexity_level": "low",
            "parameter_ranges": {"valence": "low", "arousal": "medium", ...},
            "harmonic_state": "tonic",
            "rhythmic_state": "straight",
            "suggestions": ["Consider slower tempo", ...]
        }
    """
    global _context_analyzer

    # Initialize if not already done (with double-check locking)
    if _context_analyzer is None:
        with _context_lock:
            if _context_analyzer is None:
                initialize_context_system()

    try:
        # Parse state
        state = json.loads(state_json)

        # Analyze context
        context = _context_analyzer.analyze(state)

        # Get contextual suggestions
        suggestions = _context_analyzer.get_contextual_suggestions(context)

        # Convert to JSON
        result = {
            "emotion_category": context.emotion_category,
            "complexity_level": context.complexity_level,
            "parameter_ranges": context.parameter_ranges,
            "harmonic_state": context.harmonic_state,
            "rhythmic_state": context.rhythmic_state,
            "suggestions": suggestions,
        }

        return json.dumps(result)

    except Exception as e:
        # Return default context on error
        return json.dumps({
            "emotion_category": "unknown",
            "complexity_level": "moderate",
            "parameter_ranges": {},
            "harmonic_state": None,
            "rhythmic_state": None,
            "suggestions": [],
        })


def get_contextual_parameters(state_json: str) -> str:
    """
    Get context-aware parameter adjustments.

    Args:
        state_json: Current state JSON

    Returns:
        JSON string with parameter adjustments:
        {
            "tempo": 70,
            "complexity": 0.3,
            "density": 0.2,
            "justification": "Low energy emotions benefit from slower tempos"
        }
    """
    global _context_analyzer

    if _context_analyzer is None:
        initialize_context_system()

    try:
        state = json.loads(state_json)
        context = _context_analyzer.analyze(state)

        # Generate parameter adjustments based on context
        adjustments = _generate_parameter_adjustments(context, state)

        return json.dumps(adjustments)

    except Exception as e:
        return json.dumps({})


def update_context(state_json: str):
    """
    Update context with new state information.

    Args:
        state_json: Updated state JSON
    """
    # Context analyzer is stateless, so this is a no-op
    # But we keep the function for API consistency
    pass


def get_contextual_suggestions(state_json: str) -> str:
    """
    Get contextual suggestions based on analysis.

    Args:
        state_json: Current state JSON

    Returns:
        JSON string with suggestions:
        {
            "suggestions": [
                "Consider increasing complexity for more interest",
                "Low energy emotions often benefit from slower tempos"
            ]
        }
    """
    global _context_analyzer

    if _context_analyzer is None:
        initialize_context_system()

    try:
        state = json.loads(state_json)
        context = _context_analyzer.analyze(state)
        suggestions = _context_analyzer.get_contextual_suggestions(context)

        return json.dumps({"suggestions": suggestions})

    except Exception as e:
        return json.dumps({"suggestions": []})


def _generate_parameter_adjustments(
    context: MusicalContext,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate parameter adjustments based on context.

    Args:
        context: MusicalContext object
        state: Current state dictionary

    Returns:
        Dictionary with parameter adjustments and justification
    """
    adjustments = {}
    justification_parts = []

    # Tempo adjustments based on emotion category
    if context.emotion_category == "negative_low_energy":
        adjustments["tempo"] = 70
        justification_parts.append("Low energy emotions benefit from slower tempos")
    elif context.emotion_category == "negative_high_energy":
        adjustments["tempo"] = 140
        justification_parts.append("High energy negative emotions work well with faster tempos")
    elif context.emotion_category == "positive_high_energy":
        adjustments["tempo"] = 120
        justification_parts.append("Positive high energy emotions benefit from moderate-fast tempos")
    else:
        adjustments["tempo"] = 100
        justification_parts.append("Neutral emotions work well with moderate tempos")

    # Complexity adjustments
    if context.complexity_level == "simple":
        adjustments["complexity"] = 0.3
        justification_parts.append("Simple context suggests lower complexity")
    elif context.complexity_level == "complex":
        adjustments["complexity"] = 0.7
        justification_parts.append("Complex context suggests higher complexity")
    else:
        adjustments["complexity"] = 0.5

    # Density adjustments based on emotion
    emotion = state.get("emotion", "").lower()
    if emotion in ["grief", "sadness", "longing"]:
        adjustments["density"] = 0.2
        justification_parts.append("Sad emotions benefit from sparse arrangements")
    elif emotion in ["anger", "anxiety"]:
        adjustments["density"] = 0.8
        justification_parts.append("Intense emotions benefit from dense arrangements")

    adjustments["justification"] = ". ".join(justification_parts)

    return adjustments
