"""
Groove extraction and application module.

Extract timing/velocity patterns from MIDI files and apply them to other
tracks.

Includes:
- GrooveTemplate extraction from existing MIDI
- Genre-based groove templates
- "Drunken Drummer" humanization engine for emotionally-driven processing
"""

from music_brain.groove.applicator import apply_groove, humanize
from music_brain.groove.drum_analysis import (
    DrumAnalyzer,
    DrumTechniqueProfile,
    HiHatAlternation,
    SnareBounceSignature,
    analyze_drum_technique,
)
from music_brain.groove.drum_humanizer import DrumHumanizer
from music_brain.groove.extractor import GrooveTemplate, extract_groove
from music_brain.groove.groove_engine import (
    GrooveSettings,
    get_preset,
    humanize_drums,
    humanize_midi_file,
    list_presets,
    load_presets,
    quick_humanize,
    settings_from_intent,
    settings_from_preset,
)
from music_brain.groove.templates import GENRE_TEMPLATES, get_genre_template

__all__ = [
    # Extraction
    "extract_groove",
    "GrooveTemplate",
    # Application
    "apply_groove",
    "humanize",
    # Genre templates
    "get_genre_template",
    "GENRE_TEMPLATES",
    # Drunken Drummer humanization
    "humanize_drums",
    "humanize_midi_file",
    "GrooveSettings",
    "settings_from_intent",
    "quick_humanize",
    # Preset management
    "load_presets",
    "list_presets",
    "get_preset",
    "settings_from_preset",
    # Drum analysis
    "SnareBounceSignature",
    "HiHatAlternation",
    "DrumTechniqueProfile",
    "DrumAnalyzer",
    "analyze_drum_technique",
    # Drum humanization
    "DrumHumanizer",
]
