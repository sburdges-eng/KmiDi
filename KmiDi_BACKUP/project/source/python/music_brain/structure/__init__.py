"""
Structure Analysis - Chord, section, progression analysis, and comprehensive engine.

Analyze harmonic content, detect song sections, work with
chord progressions, and run the therapy-to-music pipeline.
"""

from music_brain.structure.chord import (
    Chord,
    ChordProgression,
    analyze_chords,
    detect_key,
)
from music_brain.structure.comprehensive_engine import (
    AffectAnalyzer,
    AffectResult,
    HarmonyPlan,
    TherapySession,
    TherapyState,
    render_plan_to_midi,
    run_cli,
)
from music_brain.structure.progression import (
    diagnose_progression,
    generate_reharmonizations,
    parse_progression_string,
)
from music_brain.structure.sections import (
    Section,
    SectionType,
    detect_sections,
)
from music_brain.structure.tension_curve import (
    TENSION_CURVES,
    apply_tension_curve,
    generate_curve_for_bars,
    get_tension_curve,
    list_tension_curves,
)

__all__ = [
    # Chord analysis
    "analyze_chords",
    "ChordProgression",
    "Chord",
    "detect_key",
    # Section detection
    "detect_sections",
    "Section",
    "SectionType",
    # Progression tools
    "diagnose_progression",
    "generate_reharmonizations",
    "parse_progression_string",
    # Comprehensive engine
    "AffectAnalyzer",
    "AffectResult",
    "TherapySession",
    "TherapyState",
    "HarmonyPlan",
    "render_plan_to_midi",
    "run_cli",
    # Tension curves
    "apply_tension_curve",
    "get_tension_curve",
    "list_tension_curves",
    "generate_curve_for_bars",
    "TENSION_CURVES",
]
