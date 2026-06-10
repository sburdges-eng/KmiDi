"""
Chord Detector - Advanced chord detection with polyphonic scoring and jazz analysis.

Provides:
- ChordDetector: Basic chord detection from notes
- PolyphonicScorer: Scoring for complex polyphonic textures
- JazzChordAnalyzer: Jazz chord extensions and alterations
- ChordMatch: Chord matching result data structure
"""

from typing import List, Optional, Set
from dataclasses import dataclass
from enum import Enum

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class ChordQuality(Enum):
    """Chord quality types."""

    MAJOR = "maj"
    MINOR = "min"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    SUSPENDED_2 = "sus2"
    SUSPENDED_4 = "sus4"
    MAJOR_7 = "maj7"
    MINOR_7 = "min7"
    DOMINANT_7 = "7"
    DIMINISHED_7 = "dim7"
    HALF_DIMINISHED = "m7b5"


@dataclass
class ChordMatch:
    """Result of chord matching."""

    name: str
    root: str
    quality: ChordQuality
    bass_note: Optional[str]
    confidence: float
    intervals_matched: Set[int]

    def __str__(self):
        return self.name


class ChordDetector:
    """Detect chords from note lists."""

    def __init__(self):
        self.chord_templates = self._build_chord_templates()

    def notes_to_pitch_classes(self, notes: List[str]) -> List[int]:
        """Convert note names to pitch class indices (0-11)."""
        pitch_classes = []
        for note in notes:
            if note in NOTE_NAMES:
                pitch_classes.append(NOTE_NAMES.index(note))
        return pitch_classes

    def detect_chord(self, notes: List[str]) -> List[ChordMatch]:
        """Detect chord from list of note names."""
        if not notes:
            return []

        pitch_classes = set(self.notes_to_pitch_classes(notes))
        if not pitch_classes:
            return []

        matches = []

        # Try each root
        for root_idx in range(12):
            root_name = NOTE_NAMES[root_idx]

            # Try each quality
            for quality, intervals in self.chord_templates.items():
                chord_pcs = {(root_idx + interval) % 12 for interval in intervals}

                # Calculate match score
                intersection = pitch_classes & chord_pcs
                union = pitch_classes | chord_pcs
                confidence = len(intersection) / len(union) if union else 0.0

                if confidence > 0.3:  # Minimum threshold
                    chord_name = self._format_chord_name(root_name, quality)
                    matches.append(
                        ChordMatch(
                            name=chord_name,
                            root=root_name,
                            quality=quality,
                            bass_note=notes[0] if notes else None,
                            confidence=confidence,
                            intervals_matched=intersection,
                        )
                    )

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:5]  # Return top 5 matches

    def get_slash_chord(self, match: ChordMatch) -> str:
        """Format chord with slash notation if bass differs."""
        if match.bass_note and match.bass_note != match.root:
            return f"{match.name}/{match.bass_note}"
        return match.name

    def _build_chord_templates(self) -> dict:
        """Build chord interval templates."""
        return {
            ChordQuality.MAJOR: [0, 4, 7],
            ChordQuality.MINOR: [0, 3, 7],
            ChordQuality.DIMINISHED: [0, 3, 6],
            ChordQuality.AUGMENTED: [0, 4, 8],
            ChordQuality.SUSPENDED_2: [0, 2, 7],
            ChordQuality.SUSPENDED_4: [0, 5, 7],
            ChordQuality.MAJOR_7: [0, 4, 7, 11],
            ChordQuality.MINOR_7: [0, 3, 7, 10],
            ChordQuality.DOMINANT_7: [0, 4, 7, 10],
            ChordQuality.DIMINISHED_7: [0, 3, 6, 9],
            ChordQuality.HALF_DIMINISHED: [0, 3, 6, 10],
        }

    def _format_chord_name(self, root: str, quality: ChordQuality) -> str:
        """Format chord name."""
        if quality == ChordQuality.MAJOR:
            return root
        elif quality == ChordQuality.MINOR:
            return f"{root}m"
        elif quality == ChordQuality.DIMINISHED:
            return f"{root}dim"
        elif quality == ChordQuality.AUGMENTED:
            return f"{root}aug"
        elif quality == ChordQuality.SUSPENDED_2:
            return f"{root}sus2"
        elif quality == ChordQuality.SUSPENDED_4:
            return f"{root}sus4"
        elif quality == ChordQuality.MAJOR_7:
            return f"{root}maj7"
        elif quality == ChordQuality.MINOR_7:
            return f"{root}m7"
        elif quality == ChordQuality.DOMINANT_7:
            return f"{root}7"
        elif quality == ChordQuality.DIMINISHED_7:
            return f"{root}dim7"
        elif quality == ChordQuality.HALF_DIMINISHED:
            return f"{root}m7b5"
        return f"{root}{quality.value}"


class PolyphonicScorer:
    """Score polyphonic chord textures."""

    def score(self, pitch_classes: Set[int], chord_template: Set[int]) -> float:
        """Score how well pitch classes match chord template."""
        if not chord_template:
            return 0.0

        intersection = pitch_classes & chord_template
        union = pitch_classes | chord_template
        return len(intersection) / len(union) if union else 0.0


class JazzChordAnalyzer:
    """Analyze jazz chord extensions and alterations."""

    def analyze(self, notes: List[str]) -> dict:
        """Analyze jazz chord characteristics."""
        pitch_classes = set(ChordDetector().notes_to_pitch_classes(notes))

        # Check for extensions
        has_9th = any((pc + 2) % 12 in pitch_classes for pc in pitch_classes)
        has_11th = any((pc + 5) % 12 in pitch_classes for pc in pitch_classes)
        has_13th = any((pc + 9) % 12 in pitch_classes for pc in pitch_classes)

        return {
            "has_9th": has_9th,
            "has_11th": has_11th,
            "has_13th": has_13th,
            "complexity": len(pitch_classes),
        }
