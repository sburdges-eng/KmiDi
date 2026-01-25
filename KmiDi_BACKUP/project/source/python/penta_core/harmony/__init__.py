"""
Advanced Harmony - Sophisticated harmonic analysis and generation.

Provides:
- Jazz voicing generation
- Neo-Riemannian transformations
- Counterpoint generation
- Tension/release analysis
- Microtonal support (24-TET, just intonation)
"""

from penta_core.harmony.counterpoint import (
    CounterpointVoice,
    Species,
    check_counterpoint_rules,
    generate_counterpoint,
    get_species_rules,
)
from penta_core.harmony.jazz_voicings import (
    JazzVoicing,
    VoicingStyle,
    generate_jazz_voicing,
    get_drop_voicing,
    get_rootless_voicing,
    voice_lead_progression,
)
from penta_core.harmony.microtonal import (
    MicrotonalPitch,
    TuningSystem,
    cents_to_ratio,
    equal_temperament,
    just_intonation,
    ratio_to_cents,
)
from penta_core.harmony.neo_riemannian import (
    NeoRiemannianTransform,
    apply_transform,
    get_transform_path,
    leading_tone_transform,
    parallel_transform,
    relative_transform,
)
from penta_core.harmony.tension import (
    TensionAnalysis,
    TensionLevel,
    analyze_tension,
    plan_tension_curve,
    suggest_tension_chords,
)

__all__ = [
    # Jazz Voicings
    "JazzVoicing",
    "VoicingStyle",
    "generate_jazz_voicing",
    "voice_lead_progression",
    "get_drop_voicing",
    "get_rootless_voicing",
    # Neo-Riemannian
    "NeoRiemannianTransform",
    "apply_transform",
    "get_transform_path",
    "parallel_transform",
    "relative_transform",
    "leading_tone_transform",
    # Counterpoint
    "Species",
    "CounterpointVoice",
    "generate_counterpoint",
    "check_counterpoint_rules",
    "get_species_rules",
    # Tension
    "TensionLevel",
    "TensionAnalysis",
    "analyze_tension",
    "plan_tension_curve",
    "suggest_tension_chords",
    # Microtonal
    "TuningSystem",
    "MicrotonalPitch",
    "cents_to_ratio",
    "ratio_to_cents",
    "just_intonation",
    "equal_temperament",
]
