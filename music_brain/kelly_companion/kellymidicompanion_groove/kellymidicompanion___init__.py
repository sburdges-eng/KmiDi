"""
Groove extraction and application module.

Extract timing/velocity patterns from MIDI files and apply them to other tracks.

Includes:
- GrooveTemplate extraction from existing MIDI
- Genre-based groove templates
- "Drunken Drummer" humanization engine for emotionally-driven processing
"""

from kellymidicompanion.kellymidicompanion_groove.kellymidicompanion_extractor import (
    extract_groove,
    GrooveTemplate,
)  # noqa: E501

from kellymidicompanion.kellymidicompanion_groove.kellymidicompanion_applicator import (
    apply_groove,
    humanize,
)  # noqa: E501

from kellymidicompanion.kellymidicompanion_groove.kellymidicompanion_templates import (
    get_genre_template,
    GENRE_TEMPLATES,
)  # noqa: E501

from kellymidicompanion.kellymidicompanion_groove.kellymidicompanion_groove_engine import (
    humanize_drums,
    humanize_midi_file,
    GrooveSettings,
    settings_from_intent,
    quick_humanize,
    load_presets,
    list_presets,
    get_preset,
    settings_from_preset,
)

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
]
