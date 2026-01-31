"""
Base classes, constants, and utilities for intent processing.

This module contains:
- Music theory constants (chromatic scale, chord mappings)
- Data classes for all generated elements
- Helper functions for chord/key conversions
- ProcessorBase abstract class
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random

# =================================================================
# MUSIC THEORY CONSTANTS
# =================================================================

# Notes in chromatic order
CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHROMATIC_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Diatonic chords in major key (roman numerals)
MAJOR_DIATONIC = {
    'I': 'maj', 'ii': 'min', 'iii': 'min', 'IV': 'maj',
    'V': 'maj', 'vi': 'min', 'vii°': 'dim'
}

# Borrowed chords from parallel minor
BORROWED_FROM_MINOR = {
    'iv': 'min',      # Sad IV
    'bVI': 'maj',     # Epic chord
    'bVII': 'maj',    # Rock swagger
    'bIII': 'maj',    # Brightness from minor
    'ii°': 'dim',     # Tension
}

# Modal interchange options
MODAL_INTERCHANGE = {
    'lydian': {'#IV': 'maj'},      # Raised 4th, dreamy
    'mixolydian': {'bVII': 'maj'}, # Flat 7, rock
    'dorian': {'IV': 'maj'},       # Major IV in minor context
    'phrygian': {'bII': 'maj'},    # Flat 2, Spanish/tension
}


# =================================================================
# DATA CLASSES
# =================================================================

@dataclass
class GeneratedProgression:
    """A generated chord progression with metadata."""
    chords: List[str]
    key: str
    mode: str
    roman_numerals: List[str]
    rule_broken: str
    rule_effect: str
    voice_leading_notes: List[str] = field(default_factory=list)
    emotional_arc: List[str] = field(default_factory=list)


@dataclass
class GeneratedGroove:
    """A generated groove pattern with timing offsets."""
    pattern_name: str
    tempo_bpm: int
    swing_factor: float
    timing_offsets_16th: List[float]  # ms offset per 16th note
    velocity_curve: List[int]  # 0-127 per 16th note
    rule_broken: str
    rule_effect: str


@dataclass
class GeneratedArrangement:
    """Arrangement structure with sections."""
    sections: List[Dict]  # [{name, bars, energy, chords}]
    dynamic_arc: List[float]  # Energy per section
    rule_broken: str
    rule_effect: str


@dataclass
class GeneratedProduction:
    """Production guidelines based on intent."""
    eq_notes: List[str]
    dynamics_notes: List[str]
    space_notes: List[str]
    vocal_treatment: str
    rule_broken: str
    rule_effect: str


@dataclass
class GeneratedMelody:
    """Melodic guidelines and characteristics based on intent."""
    contour: str  # Shape of the melody (ascending, descending, arch, etc.)
    interval_character: str  # Step-wise, angular, repetitive, etc.
    phrase_structure: str  # Regular, fragmented, through-composed, etc.
    resolution_behavior: str  # Resolves, avoids resolution, hangs, etc.
    rhythmic_character: str  # Syncopated, on-beat, rubato, etc.
    range_notes: str  # Notes about melodic range
    motif_ideas: List[str] = field(default_factory=list)  # Specific melodic ideas
    performance_notes: List[str] = field(default_factory=list)
    rule_broken: str = ""
    rule_effect: str = ""


@dataclass
class GeneratedTexture:
    """Textural/timbral guidelines based on intent."""
    density_level: str  # Sparse, moderate, dense, overwhelming
    frequency_balance: str  # How frequency spectrum is filled
    element_roles: List[Dict] = field(default_factory=list)  # Role of each element
    space_character: str = ""  # Tight, wide, layered, etc.
    timbre_notes: List[str] = field(default_factory=list)
    interaction_notes: List[str] = field(default_factory=list)  # How elements interact
    rule_broken: str = ""
    rule_effect: str = ""


@dataclass
class GeneratedTemporal:
    """Temporal/time-based guidelines based on intent."""
    pacing: str  # Fast, slow, variable, etc.
    section_timing: List[Dict] = field(default_factory=list)  # Duration info per section
    pause_strategy: str = ""  # Where and how to use silence
    transition_style: str = ""  # How sections connect
    time_feel: str = ""  # Rushed, dragging, steady, elastic
    special_moments: List[Dict] = field(default_factory=list)  # Key temporal events
    rule_broken: str = ""
    rule_effect: str = ""


# =================================================================
# HELPER FUNCTIONS
# =================================================================

def _get_note_index(note: str) -> int:
    """Get chromatic index of a note."""
    note = note.replace('b', '#').upper()
    if note in CHROMATIC:
        return CHROMATIC.index(note)
    # Handle flats
    flat_to_sharp = {'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#'}
    if note in flat_to_sharp:
        return CHROMATIC.index(flat_to_sharp[note])
    return 0


def _transpose_chord(chord: str, key: str) -> str:
    """Transpose a chord to a specific key."""
    # Simple implementation - just prepend key
    root_idx = _get_note_index(key)
    return chord  # Full implementation would transpose


def _romans_to_chords(romans: List[str], key: str, mode: str) -> List[str]:
    """Convert Roman numerals to chord names in key."""
    # Simplified mapping - full implementation would be more complete
    key_root = _get_note_index(key)

    # Scale degrees for major
    major_intervals = [0, 2, 4, 5, 7, 9, 11]  # I, ii, iii, IV, V, vi, vii
    minor_intervals = [0, 2, 3, 5, 7, 8, 10]  # i, ii°, III, iv, v, VI, VII

    intervals = major_intervals if mode == "major" else minor_intervals

    result = []
    for roman in romans:
        chord = _roman_to_chord(roman, key, intervals)
        result.append(chord)

    return result


def _roman_to_chord(roman: str, key: str, intervals: List[int]) -> str:
    """Convert single Roman numeral to chord."""
    key_idx = _get_note_index(key)

    # Parse the roman numeral
    roman_clean = roman.upper().replace('5', '').replace('°', '')

    # Handle flats
    flat_offset = 0
    if roman_clean.startswith('B'):
        flat_offset = -1
        roman_clean = roman_clean[1:]

    # Map to scale degree
    degree_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}

    # Handle extensions
    suffix = ''
    for ext in ['MAJ7', 'MAJ9', 'M7', 'M9', 'ADD9', 'SUS4', 'SUS2', '7', '9', '11', '13']:
        if ext in roman.upper():
            suffix = ext.lower().replace('maj', 'maj').replace('add', 'add').replace('sus', 'sus')
            roman_clean = roman_clean.replace(ext, '')
            break

    # Get base roman
    for deg, idx in degree_map.items():
        if deg in roman_clean:
            # Calculate root note
            interval = intervals[idx] if idx < len(intervals) else 0
            root_idx = (key_idx + interval + flat_offset) % 12
            root = CHROMATIC_FLAT[root_idx] if flat_offset < 0 else CHROMATIC[root_idx]

            # Determine quality from original roman
            if roman.islower() or 'm' in roman.lower():
                quality = 'm' if '°' not in roman else 'dim'
            else:
                quality = ''

            # Handle power chords
            if '5' in roman:
                return f"{root}5"

            return f"{root}{quality}{suffix}"

    return roman  # Fallback


# =================================================================
# PROCESSOR BASE CLASS
# =================================================================

class ProcessorBase(ABC):
    """
    Abstract base class for all intent processors.

    Each processor is responsible for generating one type of musical element
    (harmony, groove, arrangement, melody, texture, temporal) based on the
    rule-breaking specified in the intent.
    """

    def __init__(self, key: str = "F", mode: str = "major", tempo: int = 120):
        """
        Initialize processor with musical parameters.

        Args:
            key: Musical key (e.g., "C", "F#", "Bb")
            mode: Musical mode (usually "major" or "minor")
            tempo: Tempo in BPM
        """
        self.key = key
        self.mode = mode
        self.tempo = tempo
        self.random = random  # Allow subclasses to use random

    @abstractmethod
    def generate(self, rule_to_break: str, **kwargs):
        """
        Generate musical element based on the rule to break.

        Args:
            rule_to_break: The specific rule being intentionally broken
            **kwargs: Additional context-specific parameters

        Returns:
            One of the Generated* dataclass instances
        """
        pass

    def _choose_variant(self, variants: List[tuple]) -> tuple:
        """
        Helper to choose a random variant from a list.

        Args:
            variants: List of (data, description) tuples

        Returns:
            Randomly selected (data, description) tuple
        """
        return random.choice(variants)
