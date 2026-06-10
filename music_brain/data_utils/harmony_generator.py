"""
Deprecated shim — forwards to ``music_brain.harmony_utils.harmony_generator``.

This module was a 99%-duplicate of ``harmony_utils/harmony_generator.py``
with minor divergences that have now been merged into the canonical
``harmony_utils`` version (key normalization for "Am"-style inputs,
mido ImportError fallback, logging instead of print).

New code should import from ``music_brain.harmony_utils.harmony_generator``.
This shim is retained so external callers do not break during the transition.
"""

from __future__ import annotations

import warnings

from music_brain.harmony_utils.harmony_generator import (  # noqa: F401
    ChordVoicing,
    HarmonyGenerator,
    HarmonyResult,
    NOTE_TO_MIDI,
    RuleBreakType,
    generate_midi_from_harmony,
)

warnings.warn(
    "music_brain.data_utils.harmony_generator is deprecated; "
    "import from music_brain.harmony_utils.harmony_generator instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ChordVoicing",
    "HarmonyGenerator",
    "HarmonyResult",
    "NOTE_TO_MIDI",
    "RuleBreakType",
    "generate_midi_from_harmony",
]
