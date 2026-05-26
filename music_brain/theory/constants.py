"""
Core musical theory constants.

Centralized source of truth for modes, pitches, scales, and MIDI note mappings.
"""

from typing import Dict, FrozenSet

# Valid modes recognized by the KmiDi generative models
VALID_MUSICAL_MODES: FrozenSet[str] = frozenset(
    ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]
)

# MIDI note numbers for base pitch classes (C-B)
NOTE_TO_MIDI: Dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def note_to_absolute_midi(name: str, octave: int = 4) -> int:
    """Convert a note name (e.g., 'C#') and an octave (e.g., 4) to an absolute MIDI pitch."""
    # Handle optional octave attached to name (e.g., 'C#4')
    if name[-1].isdigit() and len(name) > 1 and name[-2:] != "10":
        octave = int(name[-1])
        name = name[:-1]

    pitch_class = NOTE_TO_MIDI.get(name, 0)
    # MIDI octave 0 starts at C-1 or C0 depending on standard;
    # typically middle C is C4 = 60. So: 0 + (4+1)*12 = 60
    return pitch_class + ((octave + 1) * 12)
