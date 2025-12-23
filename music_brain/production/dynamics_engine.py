"""
Arrangement-aware dynamics helper.

Lightweight scaffold that turns a SongStructure and optional EmotionMatch
into per-section loudness targets and a simple automation curve. Inspired by
the Dynamics and Arrangement Guide.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from music_brain.emotion_thesaurus import EmotionMatch


@dataclass
class SongStructure:
    """Ordered list of song sections (e.g., intro, verse, chorus)."""

    sections: List[str] = field(default_factory=list)


@dataclass
class AutomationCurve:
    """Automation points as (position, level) tuples."""

    points: List[Tuple[float, float]] = field(default_factory=list)


class DynamicsEngine:
    """
    Convert emotion + structure into dynamics targets.

    This is an initial implementation; refine with guide-derived rules and
    section lengths as integration deepens.
    """

    _BASE_LEVELS = {
        "intro": "mp",
        "verse": "mp",
        "pre-chorus": "mf",
        "chorus": "f",
        "bridge": "p",
        "outro": "mp",
    }

    _LEVEL_TO_SCALAR = {
        "pp": 0.25,
        "p": 0.35,
        "mp": 0.45,
        "mf": 0.60,
        "f": 0.75,
        "ff": 0.9,
        "fff": 1.0,
    }

    def apply_section_dynamics(
        self,
        structure: SongStructure,
        emotion: Optional[EmotionMatch] = None,
    ) -> Dict[str, float]:
        """
        Return per-section dynamics (0–1 scalar).

        Rough rule: raise levels as sections progress; scale by emotion intensity.
        """
        levels: Dict[str, float] = {}
        intensity = (emotion.intensity_tier if emotion else 3) or 3
        intensity_offset = (intensity - 3) * 0.05  # small bias per tier

        for idx, section in enumerate(structure.sections):
            key = section.lower()
            base = self._LEVEL_TO_SCALAR.get(self._BASE_LEVELS.get(key, "mf"), 0.6)

            # Later sections creep up slightly to simulate energy build.
            progressive_bias = idx * 0.03
            level = base + intensity_offset + progressive_bias
            levels[section] = max(0.2, min(1.0, level))

        return levels

    def create_automation(
        self,
        structure: SongStructure,
        dynamics: Dict[str, float],
    ) -> AutomationCurve:
        """
        Build a simple step-wise automation curve from section dynamics.

        Positions are cumulative integers for now; replace with real bar/beat
        positions when structure includes timing info.
        """
        points: List[Tuple[float, float]] = []
        position = 0.0

        for section in structure.sections:
            level = dynamics.get(section, 0.6)
            points.append((position, level))
            position += 1.0

        return AutomationCurve(points=points)

