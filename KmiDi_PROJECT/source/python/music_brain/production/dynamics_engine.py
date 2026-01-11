"""
Arrangement-aware dynamics helper.

Implements guide-driven dynamics based on the Dynamics and Arrangement Guide.
Converts SongStructure and EmotionMatch into per-section loudness targets,
automation curves, and arrangement density recommendations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from music_brain.emotion.emotion_thesaurus import EmotionMatch


class DynamicLevel(Enum):
    """Musical dynamics levels from quietest to loudest."""
    PPP = ("ppp", 0.10, -24.0)  # pianississimo
    PP = ("pp", 0.20, -18.0)    # pianissimo
    P = ("p", 0.35, -12.0)      # piano
    MP = ("mp", 0.45, -9.0)     # mezzo-piano
    MF = ("mf", 0.60, -6.0)     # mezzo-forte
    F = ("f", 0.75, -3.0)       # forte
    FF = ("ff", 0.90, -1.0)     # fortissimo
    FFF = ("fff", 1.00, 0.0)    # fortississimo

    def __init__(self, name: str, scalar: float, db: float):
        self._name = name
        self.scalar = scalar
        self.db = db

    @classmethod
    def from_name(cls, name: str) -> "DynamicLevel":
        for level in cls:
            if level._name == name.lower():
                return level
        return cls.MF  # default


@dataclass
class SongStructure:
    """Ordered list of song sections with optional timing info."""
    sections: List[str] = field(default_factory=list)
    section_bars: Optional[List[int]] = None  # bars per section


@dataclass
class AutomationCurve:
    """Automation points as (bar_position, level) tuples."""
    points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class SectionDynamics:
    """Comprehensive section dynamics profile."""
    level: str = "mf"
    target_db: float = -6.0
    scalar: float = 0.6
    density: float = 0.5  # arrangement density 0-1
    notes: List[str] = field(default_factory=list)

    # Instrument guidelines
    drums: str = "full"  # minimal, light, full
    bass: str = "active"  # simple, sustained, active, driving
    guitar: str = "strumming"  # arpeggios, clean, grit, strumming, full
    keys: str = "pads"  # pads, sparse, full
    vocals: str = "full"  # solo, intimate, full, doubled, harmonies


@dataclass
class ArrangementProfile:
    """Complete arrangement dynamics for a song."""
    sections: Dict[str, SectionDynamics] = field(default_factory=dict)
    automation: AutomationCurve = field(default_factory=AutomationCurve)
    peak_section: str = ""
    quietest_section: str = ""
    contrast_ratio: float = 0.5  # ratio of quietest to loudest


class DynamicsEngine:
    """
    Convert emotion + structure into dynamics targets.

    Implements the Dynamics and Arrangement Guide rules for:
    - Per-section dynamics levels
    - Instrument density recommendations
    - Build/drop techniques
    - Contrast management
    """

    # Guide-based section defaults (from "The Pop/Rock Version" table)
    SECTION_DEFAULTS = {
        "intro": {
            "level": "pp", "to_level": "mp",
            "density": 0.2, "drums": "none", "bass": "simple",
            "guitar": "arpeggios", "keys": "pads", "vocals": "none",
            "note": "Draw listener in, set the tone"
        },
        "verse": {
            "level": "mp",
            "density": 0.4, "drums": "light", "bass": "simple",
            "guitar": "clean", "keys": "sparse", "vocals": "solo",
            "note": "Tell the story, leave room for vocals"
        },
        "verse1": {
            "level": "mp",
            "density": 0.35, "drums": "light", "bass": "simple",
            "guitar": "arpeggios", "keys": "sparse", "vocals": "intimate",
            "note": "Establishes story, quieter than V2"
        },
        "verse2": {
            "level": "mp", "to_level": "mf",
            "density": 0.45, "drums": "light", "bass": "simple",
            "guitar": "clean", "keys": "sparse", "vocals": "solo",
            "note": "Slightly fuller than V1"
        },
        "pre-chorus": {
            "level": "mf",
            "density": 0.55, "drums": "building", "bass": "active",
            "guitar": "grit", "keys": "sustained", "vocals": "building",
            "note": "Build anticipation, crescendo toward chorus"
        },
        "prechorus": {
            "level": "mf", "to_level": "f",
            "density": 0.6, "drums": "building", "bass": "active",
            "guitar": "grit", "keys": "sustained", "vocals": "building",
            "note": "Add risers, build energy"
        },
        "chorus": {
            "level": "f",
            "density": 0.75, "drums": "full", "bass": "driving",
            "guitar": "strumming", "keys": "full", "vocals": "doubled",
            "note": "Payoff, loudest section, memorable hook"
        },
        "chorus1": {
            "level": "f",
            "density": 0.7, "drums": "full", "bass": "driving",
            "guitar": "strumming", "keys": "full", "vocals": "doubled",
            "note": "First payoff"
        },
        "chorus2": {
            "level": "f", "to_level": "ff",
            "density": 0.8, "drums": "full", "bass": "driving",
            "guitar": "full", "keys": "full", "vocals": "harmonies",
            "note": "Bigger than chorus 1"
        },
        "bridge": {
            "level": "p", "to_level": "mf",
            "density": 0.3, "drums": "minimal", "bass": "sustained",
            "guitar": "clean", "keys": "pads", "vocals": "intimate",
            "note": "Contrast, break the pattern, different vibe"
        },
        "breakdown": {
            "level": "p",
            "density": 0.2, "drums": "none", "bass": "none",
            "guitar": "none", "keys": "pads", "vocals": "solo",
            "note": "Strip down for impact"
        },
        "build": {
            "level": "mf", "to_level": "ff",
            "density": 0.5, "drums": "building", "bass": "active",
            "guitar": "grit", "keys": "sustained", "vocals": "none",
            "note": "Rising energy, adding elements"
        },
        "drop": {
            "level": "ff",
            "density": 0.9, "drums": "full", "bass": "driving",
            "guitar": "full", "keys": "full", "vocals": "none",
            "note": "Maximum impact, all elements hit"
        },
        "final-chorus": {
            "level": "ff",
            "density": 0.9, "drums": "full", "bass": "driving",
            "guitar": "full", "keys": "full", "vocals": "harmonies",
            "note": "Biggest moment, add layers, consider key lift"
        },
        "finalchorus": {
            "level": "ff", "to_level": "fff",
            "density": 0.95, "drums": "full", "bass": "driving",
            "guitar": "full", "keys": "full", "vocals": "harmonies",
            "note": "Ultimate payoff, everything"
        },
        "outro": {
            "level": "mp",
            "density": 0.3, "drums": "light", "bass": "simple",
            "guitar": "arpeggios", "keys": "pads", "vocals": "fading",
            "note": "Fade or strong ending"
        },
    }

    # Emotion to dynamics modifier
    EMOTION_MODIFIERS = {
        # Negative valence, low arousal - quieter, more space
        "grief": {"level_offset": -1, "density_mod": -0.15},
        "sad": {"level_offset": -1, "density_mod": -0.1},
        "melancholy": {"level_offset": -1, "density_mod": -0.1},
        # Negative valence, high arousal - intense but controlled
        "anxiety": {"level_offset": 0, "density_mod": 0.1},
        "anger": {"level_offset": 1, "density_mod": 0.15},
        "rage": {"level_offset": 2, "density_mod": 0.2},
        # Positive valence, low arousal - gentle, spacious
        "calm": {"level_offset": -1, "density_mod": -0.2},
        "peaceful": {"level_offset": -2, "density_mod": -0.25},
        # Positive valence, high arousal - full, energetic
        "hope": {"level_offset": 0, "density_mod": 0.05},
        "joy": {"level_offset": 1, "density_mod": 0.1},
        "euphoria": {"level_offset": 2, "density_mod": 0.15},
        # Tension/build emotions
        "tension": {"level_offset": 0, "density_mod": 0.05},
        "suspense": {"level_offset": -1, "density_mod": -0.1},
    }

    LEVEL_ORDER = ["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff"]

    def _adjust_level(self, base_level: str, offset: int) -> str:
        """Shift a dynamics level by offset steps."""
        try:
            idx = self.LEVEL_ORDER.index(base_level.lower())
            new_idx = max(0, min(len(self.LEVEL_ORDER) - 1, idx + offset))
            return self.LEVEL_ORDER[new_idx]
        except ValueError:
            return base_level

    def apply_section_dynamics(
        self,
        structure: SongStructure,
        emotion: Optional[EmotionMatch] = None,
    ) -> Dict[str, float]:
        """
        Return per-section dynamics (0-1 scalar).

        Uses guide-based rules with emotion modification.
        """
        levels: Dict[str, float] = {}

        # Get emotion modifier
        emotion_key = emotion.base_emotion.lower() if emotion else "neutral"
        modifier = self.EMOTION_MODIFIERS.get(emotion_key, {"level_offset": 0})
        level_offset = modifier.get("level_offset", 0)

        # Intensity tier adjustment
        if emotion and emotion.intensity_tier:
            level_offset += (emotion.intensity_tier - 3) // 2

        for idx, section in enumerate(structure.sections):
            key = section.lower().replace(" ", "").replace("-", "")

            # Find matching default
            defaults = None
            for pattern in [key, key.rstrip("0123456789")]:
                if pattern in self.SECTION_DEFAULTS:
                    defaults = self.SECTION_DEFAULTS[pattern]
                    break

            if not defaults:
                defaults = self.SECTION_DEFAULTS.get("verse", {"level": "mf"})

            base_level = defaults.get("level", "mf")
            adjusted = self._adjust_level(base_level, level_offset)

            # Progressive build through song
            progressive_bias = idx * 0.02

            scalar = DynamicLevel.from_name(adjusted).scalar + progressive_bias
            levels[section] = max(0.1, min(1.0, scalar))

        return levels

    def apply_section_profiles(
        self,
        structure: SongStructure,
        emotion: Optional[EmotionMatch] = None,
    ) -> Dict[str, SectionDynamics]:
        """
        Return comprehensive section dynamics profiles.

        Includes arrangement density, instrument guidelines, and notes.
        """
        profiles: Dict[str, SectionDynamics] = {}

        # Get emotion modifier
        emotion_key = emotion.base_emotion.lower() if emotion else "neutral"
        modifier = self.EMOTION_MODIFIERS.get(emotion_key, {})
        level_offset = modifier.get("level_offset", 0)
        density_mod = modifier.get("density_mod", 0.0)

        if emotion and emotion.intensity_tier:
            level_offset += (emotion.intensity_tier - 3) // 2
            density_mod += (emotion.intensity_tier - 3) * 0.03

        for idx, section in enumerate(structure.sections):
            key = section.lower().replace(" ", "").replace("-", "")

            # Find matching default
            defaults = None
            for pattern in [key, key.rstrip("0123456789")]:
                if pattern in self.SECTION_DEFAULTS:
                    defaults = dict(self.SECTION_DEFAULTS[pattern])
                    break

            if not defaults:
                defaults = dict(self.SECTION_DEFAULTS.get("verse", {}))

            base_level = defaults.get("level", "mf")
            adjusted = self._adjust_level(base_level, level_offset)
            level_enum = DynamicLevel.from_name(adjusted)

            base_density = defaults.get("density", 0.5)
            adjusted_density = max(0.1, min(1.0, base_density + density_mod))

            profiles[section] = SectionDynamics(
                level=adjusted,
                target_db=level_enum.db,
                scalar=level_enum.scalar,
                density=adjusted_density,
                notes=[defaults.get("note", "")],
                drums=defaults.get("drums", "full"),
                bass=defaults.get("bass", "active"),
                guitar=defaults.get("guitar", "strumming"),
                keys=defaults.get("keys", "pads"),
                vocals=defaults.get("vocals", "full"),
            )

        return profiles

    def create_automation(
        self,
        structure: SongStructure,
        dynamics: Dict[str, float],
    ) -> AutomationCurve:
        """
        Build automation curve from section dynamics.

        Uses section bars if available, otherwise 8-bar default.
        """
        points: List[Tuple[float, float]] = []
        position = 0.0

        section_bars = structure.section_bars or [8] * len(structure.sections)

        for idx, section in enumerate(structure.sections):
            level = dynamics.get(section, 0.6)
            bars = section_bars[idx] if idx < len(section_bars) else 8

            # Start of section
            points.append((position, level))

            # If section has build (to_level), add intermediate point
            key = section.lower().replace(" ", "").replace("-", "")
            defaults = self.SECTION_DEFAULTS.get(key, {})
            if "to_level" in defaults:
                end_level = DynamicLevel.from_name(defaults["to_level"]).scalar
                # Ramp up through section
                mid_point = position + bars * 0.5
                points.append((mid_point, (level + end_level) / 2))
                points.append((position + bars - 0.5, end_level))

            position += bars

        return AutomationCurve(points=points)

    def get_arrangement_profile(
        self,
        structure: SongStructure,
        emotion: Optional[EmotionMatch] = None,
    ) -> ArrangementProfile:
        """
        Get complete arrangement profile for a song.

        Includes all section dynamics, automation, and contrast analysis.
        """
        profiles = self.apply_section_profiles(structure, emotion)
        scalar_levels = {s: p.scalar for s, p in profiles.items()}

        automation = self.create_automation(structure, scalar_levels)

        # Find peak and quietest
        peak_section = max(scalar_levels, key=lambda s: scalar_levels[s])
        quietest_section = min(scalar_levels, key=lambda s: scalar_levels[s])

        peak_level = scalar_levels[peak_section]
        quiet_level = scalar_levels[quietest_section]
        contrast = quiet_level / peak_level if peak_level > 0 else 0.5

        return ArrangementProfile(
            sections=profiles,
            automation=automation,
            peak_section=peak_section,
            quietest_section=quietest_section,
            contrast_ratio=contrast,
        )

    def suggest_contrast_improvements(
        self,
        profile: ArrangementProfile,
    ) -> List[str]:
        """
        Analyze arrangement and suggest improvements based on guide rules.
        """
        suggestions = []

        # Check contrast ratio (guide says quietest should be ~50% of loudest)
        if profile.contrast_ratio > 0.7:
            suggestions.append(
                f"Low contrast ({profile.contrast_ratio:.0%}). "
                f"Strip down '{profile.quietest_section}' more for impact."
            )

        # Check if chorus is actually loudest
        for section, dynamics in profile.sections.items():
            if "chorus" in section.lower() and section != profile.peak_section:
                suggestions.append(
                    f"'{section}' should typically be loudest. "
                    f"Currently peak is '{profile.peak_section}'."
                )
                break

        # Check for build before chorus
        sections_list = list(profile.sections.keys())
        for i, section in enumerate(sections_list):
            if "chorus" in section.lower() and i > 0:
                prev = sections_list[i - 1]
                prev_level = profile.sections[prev].scalar
                chorus_level = profile.sections[section].scalar
                if chorus_level - prev_level < 0.1:
                    suggestions.append(
                        f"Add more build before '{section}'. "
                        f"Consider adding pre-chorus or risers."
                    )

        # Check final chorus is biggest
        final_chorus = [s for s in sections_list if "final" in s.lower() and "chorus" in s.lower()]
        if final_chorus:
            fc = final_chorus[0]
            if fc != profile.peak_section:
                suggestions.append(
                    f"'{fc}' should be the biggest moment. "
                    f"Add more elements, consider key lift."
                )

        return suggestions
