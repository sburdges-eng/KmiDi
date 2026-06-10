"""
Key Analyzer - Key detection and scale analysis.

Provides:
- KeyAnalyzer: Detect key from pitch classes
- KeyScaleAnalyzer: Analyze chords in key context
- KeyEstimate: Key detection result
- Mode: Musical modes
- ModalInterchangeDetector: Detect borrowed chords
- SecondaryDominantDetector: Detect secondary dominants
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class Mode(Enum):
    """Musical modes."""

    IONIAN = "ionian"  # Major
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"  # Natural minor
    LOCRIAN = "locrian"


@dataclass
class KeyEstimate:
    """Key detection result."""

    root: str
    mode: Mode
    confidence: float
    pitch_class: int  # 0-11

    def __str__(self):
        mode_str = "major" if self.mode == Mode.IONIAN else "minor"
        return f"{self.root} {mode_str}"


class KeyAnalyzer:
    """Detect key from pitch class distribution."""

    # Major scale intervals
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

    def detect_key(self, pitch_classes: List[int]) -> List[KeyEstimate]:
        """Detect key from pitch class list."""
        if not pitch_classes:
            return []

        # Count occurrences
        pc_counts = {}
        for pc in pitch_classes:
            pc_counts[pc] = pc_counts.get(pc, 0) + 1

        estimates = []

        # Try each key
        for root_idx in range(12):
            root_name = NOTE_NAMES[root_idx]

            # Major
            major_pcs = {(root_idx + interval) % 12 for interval in self.MAJOR_SCALE}
            major_score = sum(pc_counts.get(pc, 0) for pc in major_pcs) / len(pitch_classes)

            # Minor
            minor_pcs = {(root_idx + interval) % 12 for interval in self.MINOR_SCALE}
            minor_score = sum(pc_counts.get(pc, 0) for pc in minor_pcs) / len(pitch_classes)

            if major_score > 0.3:
                estimates.append(
                    KeyEstimate(
                        root=root_name,
                        mode=Mode.IONIAN,
                        confidence=major_score,
                        pitch_class=root_idx,
                    )
                )

            if minor_score > 0.3:
                estimates.append(
                    KeyEstimate(
                        root=root_name,
                        mode=Mode.AEOLIAN,
                        confidence=minor_score,
                        pitch_class=root_idx,
                    )
                )

        # Sort by confidence
        estimates.sort(key=lambda e: e.confidence, reverse=True)
        return estimates[:3]  # Return top 3


class KeyScaleAnalyzer:
    """Analyze chords in key context."""

    # Roman numeral mapping for major key
    MAJOR_ROMAN = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii°"}

    # Roman numeral mapping for minor key
    MINOR_ROMAN = {0: "i", 2: "ii°", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}

    def analyze_chord_in_key(
        self,
        root_pc: int,
        quality: str,
        key: KeyEstimate,
        prev_chord: Optional[Tuple[int, str]] = None,
        next_chord: Optional[Tuple[int, str]] = None,
    ) -> Dict:
        """Analyze chord in key context."""
        # Calculate relative position
        relative_pc = (root_pc - key.pitch_class) % 12

        # Get roman numeral
        if key.mode == Mode.IONIAN:
            roman = self.MAJOR_ROMAN.get(relative_pc, "?")
        else:
            roman = self.MINOR_ROMAN.get(relative_pc, "?")

        # Determine function
        function_map = {
            "I": "tonic",
            "i": "tonic",
            "V": "dominant",
            "v": "dominant",
            "IV": "subdominant",
            "iv": "subdominant",
            "ii": "subdominant",
            "ii°": "subdominant",
            "vi": "submediant",
            "VI": "submediant",
            "iii": "mediant",
            "III": "mediant",
        }
        function = function_map.get(roman, "unknown")

        return {"roman_numeral": roman, "function": function, "in_key": True}


class ModalInterchangeDetector:
    """Detect modal interchange (borrowed chords)."""

    def detect(self, chord_pc: int, key: KeyEstimate) -> Optional[Dict]:
        """Detect if chord is borrowed from parallel mode."""
        # Simple implementation - check if chord is in parallel mode but not current mode
        parallel_major_pcs = {
            (key.pitch_class + interval) % 12 for interval in KeyAnalyzer.MAJOR_SCALE
        }
        parallel_minor_pcs = {
            (key.pitch_class + interval) % 12 for interval in KeyAnalyzer.MINOR_SCALE
        }

        if key.mode == Mode.IONIAN:
            # Check if chord is from parallel minor
            if chord_pc in parallel_minor_pcs and chord_pc not in parallel_major_pcs:
                return {"borrowed_from": "minor", "chord_pc": chord_pc}
        else:
            # Check if chord is from parallel major
            if chord_pc in parallel_major_pcs and chord_pc not in parallel_minor_pcs:
                return {"borrowed_from": "major", "chord_pc": chord_pc}

        return None


class SecondaryDominantDetector:
    """Detect secondary dominants."""

    def detect(self, chord_pc: int, key: KeyEstimate) -> Optional[Dict]:
        """Detect if chord is a secondary dominant."""
        # Secondary dominants are V of any scale degree
        # For now, simple check: is it a dominant 7th chord?
        # More sophisticated: check if it resolves to a chord in the key

        # This is a simplified implementation
        # Full implementation would check resolution patterns
        return None
