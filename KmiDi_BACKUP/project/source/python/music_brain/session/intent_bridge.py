"""
Intent Bridge - Python interface for C++ intent processing.

Provides functions to process intents using Python intent_processor and convert
between Python CompleteSongIntent and C++ IntentResult formats.
"""

from typing import Dict, List, Any, Optional
import json

from music_brain.session.intent_processor import IntentProcessor
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    HarmonyRuleBreak,
    RhythmRuleBreak,
    ArrangementRuleBreak,
    ProductionRuleBreak,
    MelodyRuleBreak,
)


# Global instance (singleton pattern)
_intent_processor: Optional[IntentProcessor] = None


def initialize_intent_system():
    """Initialize the intent processing system."""
    global _intent_processor
    if _intent_processor is None:
        _intent_processor = IntentProcessor()


def process_intent(intent_json: str) -> str:
    """
    Process intent using Python intent_processor.

    This function is designed to be called from C++ via Python bridge.

    Args:
        intent_json: JSON string with Python CompleteSongIntent format:
            {
                "phase_0": {
                    "core_event": "...",
                    "core_resistance": "...",
                    "core_longing": "..."
                },
                "phase_1": {
                    "mood_primary": "grief",
                    "vulnerability_scale": 0.8,
                    ...
                },
                "phase_2": {
                    "technical_genre": "ambient",
                    "technical_key": "C",
                    "technical_rule_to_break": ["HARMONY_AvoidTonicResolution"],
                    ...
                }
            }

    Returns:
        JSON string with C++ IntentResult format:
        {
            "key": "C",
            "mode": "minor",
            "tempoBpm": 82,
            "chordProgression": ["Am", "Dm", "F", "C"],
            "ruleBreaks": [...],
            "melodicRange": 0.6,
            ...
        }
    """
    global _intent_processor

    # Initialize if not already done
    if _intent_processor is None:
        initialize_intent_system()

    try:
        # Parse intent
        intent_dict = json.loads(intent_json)
        intent = CompleteSongIntent.from_dict(intent_dict)

        # Process intent
        result = _intent_processor.process_intent(intent)

        # Convert to C++ format
        cpp_result = _convert_to_cpp_format(result)

        return json.dumps(cpp_result)

    except Exception as e:
        # Return default result on error
        return json.dumps(_get_default_cpp_result())


def convert_to_cpp_intent(intent_json: str) -> str:
    """
    Convert Python CompleteSongIntent to C++ IntentResult format.

    Args:
        intent_json: Python intent JSON

    Returns:
        JSON string with C++ IntentResult format
    """
    return process_intent(intent_json)


def convert_to_python_intent(cpp_intent_json: str) -> str:
    """
    Convert C++ IntentResult to Python CompleteSongIntent format.

    Args:
        cpp_intent_json: C++ IntentResult JSON

    Returns:
        JSON string with Python intent format
    """
    try:
        cpp_intent = json.loads(cpp_intent_json)

        # Convert to Python format
        python_intent = {
            "phase_1": {
                "mood_primary": cpp_intent.get("emotion", "neutral"),
                "vulnerability_scale": 0.5,
            },
            "phase_2": {
                "technical_key": cpp_intent.get("key", "C"),
                "technical_mode": cpp_intent.get("mode", "major"),
                "technical_rule_to_break": cpp_intent.get("ruleBreaks", []),
            },
        }

        return json.dumps(python_intent)

    except Exception as e:
        return json.dumps({
            "phase_1": {"mood_primary": "neutral"},
            "phase_2": {"technical_key": "C", "technical_mode": "major"},
        })


def validate_result(result_json: str) -> bool:
    """
    Validate intent processing result.

    Args:
        result_json: Result JSON from Python

    Returns:
        True if result is valid
    """
    try:
        result = json.loads(result_json)

        # Basic validation: check required fields
        required_fields = ["key", "mode", "tempoBpm"]
        for field in required_fields:
            if field not in result:
                return False

        # Validate ranges
        if not (20 <= result.get("tempoBpm", 0) <= 300):
            return False

        return True

    except Exception as e:
        return False


def get_suggested_rule_breaks(emotion: str) -> str:
    """
    Get suggested rule breaks for an emotion.

    Args:
        emotion: Emotion name (e.g., "grief", "longing")

    Returns:
        JSON string with suggested rule breaks:
        {
            "rule_breaks": ["HARMONY_AvoidTonicResolution", ...],
            "justifications": {...}
        }
    """
    from music_brain.session.intent_schema import RULE_BREAKING_EFFECTS

    emotion_lower = emotion.lower()
    suggested_breaks = []
    justifications = {}

    # Map emotions to rule breaks
    emotion_to_rule_breaks = {
        "grief": ["HARMONY_AvoidTonicResolution", "HARMONY_ModalInterchange"],
        "longing": ["HARMONY_UnresolvedDissonance", "MELODY_AvoidResolution"],
        "anger": ["RHYTHM_ConstantDisplacement", "PRODUCTION_Distortion"],
        "anxiety": ["RHYTHM_ConstantDisplacement", "HARMONY_UnresolvedDissonance"],
        "hope": ["HARMONY_ModalInterchange", "MELODY_AntiClimax"],
    }

    rule_breaks = emotion_to_rule_breaks.get(emotion_lower, [])

    for rule_break in rule_breaks:
        if rule_break in RULE_BREAKING_EFFECTS:
            suggested_breaks.append(rule_break)
            justifications[rule_break] = RULE_BREAKING_EFFECTS[rule_break].get(
                "justification", ""
            )

    return json.dumps({
        "rule_breaks": suggested_breaks,
        "justifications": justifications,
    })


def _convert_to_cpp_format(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Python intent processor result to C++ IntentResult format.

    Args:
        result: Python intent processor result

    Returns:
        Dictionary in C++ IntentResult format
    """
    cpp_result = {
        "key": result.get("key", "C"),
        "mode": result.get("mode", "major"),
        "tempoBpm": result.get("tempo", 120),
        "chordProgression": result.get("chords", []),
        "ruleBreaks": result.get("rule_breaks", []),
        "melodicRange": result.get("melodic_range", 0.6),
        "leapProbability": result.get("leap_probability", 0.3),
        "allowChromaticism": result.get("allow_chromaticism", False),
        "swingAmount": result.get("swing_amount", 0.0),
        "syncopationLevel": result.get("syncopation_level", 0.3),
        "humanization": result.get("humanization", 0.15),
        "baseVelocity": result.get("base_velocity", 0.6),
        "dynamicRange": result.get("dynamic_range", 0.4),
    }

    return cpp_result


def _get_default_cpp_result() -> Dict[str, Any]:
    """Get default C++ result when processing fails."""
    return {
        "key": "C",
        "mode": "major",
        "tempoBpm": 120,
        "chordProgression": [],
        "ruleBreaks": [],
        "melodicRange": 0.6,
        "leapProbability": 0.3,
        "allowChromaticism": False,
        "swingAmount": 0.0,
        "syncopationLevel": 0.3,
        "humanization": 0.15,
        "baseVelocity": 0.6,
        "dynamicRange": 0.4,
    }
