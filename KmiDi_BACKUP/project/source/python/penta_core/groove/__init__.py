"""
Advanced Groove - Sophisticated rhythm analysis and manipulation.

Provides:
- Polyrhythm detection and generation
- Groove DNA extraction (timing fingerprint)
- Humanization presets by artist/style
- Live performance timing analysis
- Drum replacement with timing preservation
"""

from penta_core.groove.drum_replacement import (
    DrumReplacement,
    get_drum_samples,
    map_drum_hits,
    preserve_timing,
    replace_drums,
)
from penta_core.groove.groove_dna import (
    GrooveDNA,
    apply_groove_dna,
    compare_grooves,
    extract_groove_dna,
    get_artist_groove_dna,
)
from penta_core.groove.humanization import (
    HumanizationPreset,
    HumanizationStyle,
    create_custom_preset,
    get_artist_preset,
    get_genre_preset,
    humanize_midi,
)
from penta_core.groove.performance import (
    PerformanceAnalysis,
    TimingProfile,
    analyze_live_performance,
    detect_tempo_variations,
    extract_expression,
)
from penta_core.groove.polyrhythm import (
    Polyrhythm,
    PolyrhythmPattern,
    calculate_lcm_duration,
    detect_polyrhythm,
    generate_polyrhythm,
    get_common_polyrhythms,
)

__all__ = [
    # Polyrhythm
    "Polyrhythm",
    "PolyrhythmPattern",
    "detect_polyrhythm",
    "generate_polyrhythm",
    "get_common_polyrhythms",
    "calculate_lcm_duration",
    # Groove DNA
    "GrooveDNA",
    "extract_groove_dna",
    "compare_grooves",
    "apply_groove_dna",
    "get_artist_groove_dna",
    # Humanization
    "HumanizationPreset",
    "HumanizationStyle",
    "humanize_midi",
    "get_artist_preset",
    "get_genre_preset",
    "create_custom_preset",
    # Performance
    "PerformanceAnalysis",
    "TimingProfile",
    "analyze_live_performance",
    "detect_tempo_variations",
    "extract_expression",
    # Drum Replacement
    "DrumReplacement",
    "replace_drums",
    "map_drum_hits",
    "preserve_timing",
    "get_drum_samples",
]
