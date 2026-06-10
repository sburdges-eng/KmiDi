"""
Harmony Engine - Probabilistic harmony generation and analysis.

Provides:
- ProbabilisticHarmonyEngine: Generate chord progressions
- Genre: Musical genres
- CadenceDetector: Detect cadences
- VoiceLeadingAnalyzer: Analyze voice leading
- TensionResolutionAnalyzer: Analyze tension/resolution
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Genre(Enum):
    """Musical genres."""

    POP = "pop"
    ROCK = "rock"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    BLUES = "blues"
    COUNTRY = "country"
    ELECTRONIC = "electronic"
    FOLK = "folk"


@dataclass
class ChordPrediction:
    """Predicted next chord."""

    chord: str
    score: float
    reasons: List[str]


class ProbabilisticHarmonyEngine:
    """Generate chord progressions based on genre and context."""

    # Common progressions by genre
    PROGRESSIONS = {
        Genre.POP: [
            ["I", "V", "vi", "IV"],  # 50s progression
            ["vi", "IV", "I", "V"],  # Pop punk
            ["I", "vi", "IV", "V"],  # Classic pop
        ],
        Genre.ROCK: [
            ["I", "IV", "V", "I"],
            ["I", "bVII", "IV", "I"],
            ["i", "bVII", "bVI", "bVII"],
        ],
        Genre.JAZZ: [
            ["ii", "V", "I"],
            ["iii", "vi", "ii", "V"],
            ["I", "vi", "ii", "V"],
        ],
    }

    def __init__(self, genre: Genre = Genre.POP):
        self.genre = genre
        self.cadence = CadenceDetector()

    def suggest_next_chord(
        self, progression: List[str], prefer_resolution: bool = False
    ) -> List[Dict]:
        """Suggest next chord based on progression."""
        if not progression:
            return [{"chord": "I", "score": 1.0, "reasons": ["start"]}]

        # Get genre-specific progressions
        genre_progs = self.PROGRESSIONS.get(self.genre, self.PROGRESSIONS[Genre.POP])

        suggestions = []

        # Find matching progressions
        for prog in genre_progs:
            for i in range(len(prog)):
                if progression == prog[: i + 1]:
                    if i + 1 < len(prog):
                        next_chord = prog[i + 1]
                        suggestions.append(
                            {
                                "chord": next_chord,
                                "score": 0.8,
                                "reasons": [f"follows {self.genre.value} pattern"],
                            }
                        )

        # Default suggestions if no match
        if not suggestions:
            last = progression[-1] if progression else "I"

            # Common resolutions
            resolutions = {
                "V": ["I", "vi"],
                "vi": ["IV", "V"],
                "IV": ["I", "V"],
                "ii": ["V", "vi"],
            }

            for chord, score in resolutions.get(last, [("I", 0.5)]):
                suggestions.append(
                    {"chord": chord, "score": score, "reasons": ["common resolution"]}
                )

        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:5]


class CadenceDetector:
    """Detect cadences in chord progressions."""

    CADENCE_PATTERNS = {
        "authentic": ["V", "I"],
        "plagal": ["IV", "I"],
        "deceptive": ["V", "vi"],
        "half": ["I", "V"],
    }

    def detect_cadence(self, progression: List[str]) -> Optional[Dict]:
        """Detect cadence at end of progression."""
        if len(progression) < 2:
            return None

        last_two = progression[-2:]

        for name, pattern in self.CADENCE_PATTERNS.items():
            if last_two == pattern:
                return {"name": name, "chords": last_two, "strength": 1.0}

        return None


class VoiceLeadingAnalyzer:
    """Analyze voice leading between chords."""

    def voice_leading_score(self, prev_voicing: List[int], curr_voicing: List[int]) -> float:
        """Score voice leading quality (0-1, higher is better)."""
        if not prev_voicing or not curr_voicing:
            return 0.5

        # Calculate average voice leading distance
        # Lower distance = better voice leading
        total_distance = 0
        count = 0

        for prev_pc in prev_voicing:
            min_distance = min(abs((curr_pc - prev_pc) % 12) for curr_pc in curr_voicing)
            total_distance += min_distance
            count += 1

        avg_distance = total_distance / count if count > 0 else 6.0

        # Convert to score (0-1, where 0 distance = 1.0 score)
        score = max(0.0, 1.0 - (avg_distance / 6.0))
        return score

    def suggest_voicing(self, target_pcs: Set[int], prev_voicing: List[int]) -> List[int]:
        """Suggest optimal voicing for target chord."""
        if not target_pcs:
            return []

        # Simple: use target pitch classes, prefer close voicing
        target_list = sorted(target_pcs)

        # If we have previous voicing, try to minimize movement
        if prev_voicing:
            result = []
            for prev_pc in prev_voicing:
                # Find closest target PC
                closest = min(target_list, key=lambda pc: abs((pc - prev_pc) % 12))
                result.append(closest)
            return result

        return list(target_pcs)


class TensionResolutionAnalyzer:
    """Analyze tension and resolution in harmony."""

    TENSION_CHORDS = {"V", "vii°", "ii", "iv"}
    RESOLUTION_CHORDS = {"I", "i", "vi", "VI"}

    def chord_tension(self, intervals: Set[int]) -> float:
        """Calculate tension level of chord (0-1)."""
        # Diminished and augmented intervals create tension
        tension_intervals = {1, 2, 6, 10}  # Semitones, whole tones, tritones

        tension_count = sum(1 for ivl in intervals if ivl in tension_intervals)
        return min(1.0, tension_count / len(intervals)) if intervals else 0.0

    def is_resolution(self, prev_chord: str, curr_chord: str) -> bool:
        """Check if progression is a resolution."""
        return prev_chord in self.TENSION_CHORDS and curr_chord in self.RESOLUTION_CHORDS
