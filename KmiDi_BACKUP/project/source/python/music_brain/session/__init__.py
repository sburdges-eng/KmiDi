"""
Session - Song generation, teaching modules, and interactive tools.

Interactive teaching for music theory and production concepts.
Interrogation-first songwriting assistance.
Intent-based generation with rule-breaking support.
ML-based melody generation with emotion-aware patterns.
"""

from music_brain.session.intent_schema import (
    RULE_BREAKING_EFFECTS,
    ArrangementRuleBreak,
    CompleteSongIntent,
    CoreStakes,
    GrooveFeel,
    HarmonyRuleBreak,
    NarrativeArc,
    ProductionRuleBreak,
    RhythmRuleBreak,
    SongIntent,
    SongRoot,
    SystemDirective,
    TechnicalConstraints,
    VulnerabilityScale,
    get_rule_breaking_info,
    list_all_rules,
    suggest_rule_break,
    validate_intent,
)
from music_brain.session.interrogator import SongInterrogator
from music_brain.session.ml_melody_generator import (
    GeneratedMelody,
    MelodyConfig,
    MLMelodyGenerator,
    create_melody_generator,
    generate_melody,
)
from music_brain.session.teaching import RuleBreakingTeacher

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
    # Melody Generation
    "MLMelodyGenerator",
    "MelodyConfig",
    "GeneratedMelody",
    "create_melody_generator",
    "generate_melody",
]
