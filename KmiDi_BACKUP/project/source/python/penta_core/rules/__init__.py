"""
Penta Core Rules Package
========================

Comprehensive music theory rules with context-dependent severity.
"""

from .base import Rule, RuleBreakSuggestion, RuleViolation
from .context import CONTEXT_GROUPS, MusicalContext
from .counterpoint_rules import CounterpointRules
from .emotion import Emotion, get_emotions_for_technique, get_techniques_for_emotion
from .harmony_rules import HarmonyRules
from .rhythm_rules import RhythmRules
from .severity import RuleSeverity
from .species import Species
from .timing import SwingType, TimingPocket, apply_pocket_to_midi, get_genre_pocket
from .voice_leading import VoiceLeadingRules

__all__ = [
    # Enums
    "RuleSeverity",
    "Species",
    "MusicalContext",
    "CONTEXT_GROUPS",
    "Emotion",
    "SwingType",
    # Base classes
    "Rule",
    "RuleViolation",
    "RuleBreakSuggestion",
    "TimingPocket",
    # Rule collections
    "VoiceLeadingRules",
    "HarmonyRules",
    "CounterpointRules",
    "RhythmRules",
    # Functions
    "get_techniques_for_emotion",
    "get_emotions_for_technique",
    "get_genre_pocket",
    "apply_pocket_to_midi",
]
