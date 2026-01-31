"""
Harmony module - Generate chord progressions and voicings from emotional intent.

This module translates emotional/creative intent into harmonic structures
with intentional rule-breaking applied.
"""

from music_brain.harmony.harmony_generator import (
    HarmonyGenerator,
    HarmonyResult,
    ChordVoicing,
    RuleBreakType,
    generate_midi_from_harmony
)
__all__ = [
    'HarmonyGenerator',
    'HarmonyResult',
    'ChordVoicing',
    'RuleBreakType',
    'generate_midi_from_harmony'
]
