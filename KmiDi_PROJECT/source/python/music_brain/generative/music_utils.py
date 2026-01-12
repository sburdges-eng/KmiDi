"""
Musical utilities for generative models.

Provides common functions for note conversion, scale calculations,
and musical constants used across all generative modules.
"""

from typing import List, Dict, Optional, Tuple

# Chromatic note names (using flats for consistency)
CHROMATIC_NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Enharmonic equivalents
ENHARMONIC_MAP = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"
}

# Note to pitch class mapping (0-11)
NOTE_TO_PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0
}

# Scale interval patterns
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "chromatic": list(range(12)),
}


def note_to_midi(note: str, octave: int = 4) -> int:
    """
    Convert a note name to MIDI note number.
    
    Args:
        note: Note name (e.g., "C", "F#", "Bb")
        octave: Octave number (4 = middle octave, C4 = MIDI 60)
        
    Returns:
        MIDI note number (0-127)
        
    Examples:
        >>> note_to_midi("C", 4)
        60
        >>> note_to_midi("A", 4)
        69
        >>> note_to_midi("C", 2)
        36
    """
    # Parse note name
    if len(note) > 1 and note[1] in "#b":
        base = note[:2]
    else:
        base = note[0]
    
    # Handle enharmonic equivalents
    base = ENHARMONIC_MAP.get(base, base)
    
    pitch_class = NOTE_TO_PITCH_CLASS.get(base, 0)
    return 12 * (octave + 1) + pitch_class


def midi_to_note(midi: int) -> Tuple[str, int]:
    """
    Convert MIDI note number to note name and octave.
    
    Args:
        midi: MIDI note number (0-127)
        
    Returns:
        Tuple of (note_name, octave)
        
    Examples:
        >>> midi_to_note(60)
        ('C', 4)
        >>> midi_to_note(69)
        ('A', 4)
    """
    octave = (midi // 12) - 1
    pitch_class = midi % 12
    note = CHROMATIC_NOTES[pitch_class]
    return note, octave


def get_scale_notes(root: str, scale: str = "major", octave: int = 4, num_octaves: int = 2) -> List[int]:
    """
    Get MIDI note numbers for a scale.
    
    Args:
        root: Root note name
        scale: Scale type (major, minor, dorian, etc.)
        octave: Starting octave
        num_octaves: Number of octaves to generate
        
    Returns:
        List of MIDI note numbers in the scale
    """
    root_midi = note_to_midi(root, octave)
    intervals = SCALE_INTERVALS.get(scale.lower(), SCALE_INTERVALS["major"])
    
    notes = []
    for oct_offset in range(num_octaves):
        for interval in intervals:
            note = root_midi + oct_offset * 12 + interval
            if 0 <= note <= 127:
                notes.append(note)
    
    return sorted(set(notes))


def transpose_note(note: str, semitones: int) -> str:
    """
    Transpose a note by a number of semitones.
    
    Args:
        note: Note name (e.g., "C", "F#")
        semitones: Number of semitones to transpose (positive = up, negative = down)
        
    Returns:
        Transposed note name
    """
    # Get pitch class
    if len(note) > 1 and note[1] in "#b":
        base = note[:2]
        suffix = note[2:]
    else:
        base = note[0]
        suffix = note[1:] if len(note) > 1 else ""
    
    base = ENHARMONIC_MAP.get(base, base)
    
    pitch_class = NOTE_TO_PITCH_CLASS.get(base, 0)
    new_pitch_class = (pitch_class + semitones) % 12
    new_note = CHROMATIC_NOTES[new_pitch_class]
    
    return new_note + suffix


def is_minor_key(key: str) -> bool:
    """Check if a key is minor."""
    return "m" in key and "maj" not in key


def get_chord_root(chord: str) -> str:
    """Extract the root note from a chord symbol."""
    if len(chord) > 1 and chord[1] in "b#":
        return chord[:2]
    return chord[0] if chord else "C"


def get_chord_quality(chord: str) -> str:
    """Extract the quality from a chord symbol."""
    root = get_chord_root(chord)
    return chord[len(root):]


def find_note_in_scale(note: int, scale_notes: List[int]) -> int:
    """
    Find the index of a note in a scale, or the closest note.
    
    Args:
        note: MIDI note number
        scale_notes: List of MIDI notes in the scale
        
    Returns:
        Index in scale_notes
    """
    if not scale_notes:
        return 0
    
    # Try exact match first
    if note in scale_notes:
        return scale_notes.index(note)
    
    # Find closest note
    closest_idx = 0
    min_distance = abs(note - scale_notes[0])
    
    for i, scale_note in enumerate(scale_notes):
        distance = abs(note - scale_note)
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
    
    return closest_idx


def clamp_note_to_range(note: int, min_note: int = 36, max_note: int = 96) -> int:
    """
    Clamp a MIDI note to a valid range, adjusting by octaves.
    
    Args:
        note: MIDI note number
        min_note: Minimum allowed note
        max_note: Maximum allowed note
        
    Returns:
        Note adjusted to be within range
    """
    while note < min_note:
        note += 12
    while note > max_note:
        note -= 12
    return note
