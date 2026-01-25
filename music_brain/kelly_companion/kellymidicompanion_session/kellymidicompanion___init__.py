"""
Session - Song generation, teaching modules, and interactive tools.

Interactive teaching for music theory and production concepts.
Interrogation-first songwriting assistance.
Intent-based generation with rule-breaking support.
"""

from kellymidicompanion.kellymidicompanion_session.kellymidicompanion_teaching import RuleBreakingTeacher
from kellymidicompanion.kellymidicompanion_session.kellymidicompanion_interrogator import SongInterrogator
from kellymidicompanion.kellymidicompanion_session.kellymidicompanion_intent_schema import (
    CompleteSongIntent,
    SongRoot,
    SongIntent,
    TechnicalConstraints,
    SystemDirective,
    HarmonyRuleBreak,
    RhythmRuleBreak,
    ArrangementRuleBreak,
    ProductionRuleBreak,
    VulnerabilityScale,
    NarrativeArc,
    CoreStakes,
    GrooveFeel,
    suggest_rule_break,
    get_rule_breaking_info,
    validate_intent,
    list_all_rules,
    RULE_BREAKING_EFFECTS,
)

__all__ = [
    # Teaching
    "RuleBreakingTeacher",
    # Interrogation
    "SongInterrogator",
    # Intent Schema
    "CompleteSongIntent",
    "SongRoot",
    "SongIntent",
    "TechnicalConstraints",
    "SystemDirective",
    # Rule Breaking Enums
    "HarmonyRuleBreak",
    "RhythmRuleBreak",
    "ArrangementRuleBreak",
    "ProductionRuleBreak",
    "VulnerabilityScale",
    "NarrativeArc",
    "CoreStakes",
    "GrooveFeel",
    # Functions
    "suggest_rule_break",
    "get_rule_breaking_info",
    "validate_intent",
    "list_all_rules",
    "RULE_BREAKING_EFFECTS",
]
