"""
Emotion → production mapper.

Encodes a compact, code-first version of the Production_Workflows guides
(drums, dynamics, arrangement, tempo/feel, transitions) so callers can request
guide-informed `ProductionPreset` objects directly from an `EmotionMatch`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, cast

from music_brain.emotion.emotion_thesaurus import EmotionMatch


@dataclass
class ProductionPreset:
    """Container for production decisions derived from an emotion."""

    drum_style: str = "standard"          # e.g., rock, hip-hop, jazzy
    dynamics_level: str = "mf"            # pp → fff
    arrangement_density: float = 0.5      # 0–1, sparse to dense
    intensity_tier: Optional[int] = None  # 1–6 from EmotionMatch
    tempo_range: Tuple[int, int] = (100, 120)
    feel: str = "straight"                # straight or swing
    swing: float = 0.0                    # 0–0.25
    groove_motif: str = "backbeat"
    kit_hint: str = "standard kit"
    section_dynamics: Dict[str, str] = field(default_factory=dict)
    section_density: Dict[str, float] = field(default_factory=dict)
    fx: Dict[str, str] = field(default_factory=dict)
    transitions: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


class EmotionProductionMapper:
    """
    Map EmotionMatch → ProductionPreset.

    Defaults lean on common guide cues:
    - happy/surprise → brighter, denser, pop/edm-friendly drums
    - sad/fear → sparser, laid-back, acoustic/jazzy drums
    - angry/disgust → aggressive, tight, rock/industrial drums
    """

    _NEUTRAL_PROFILE = {
        "drum_style": "standard",
        "density": 0.5,
        "tempo_range": (100, 120),
        "feel": "straight",
        "swing": 0.02,
        "groove": "backbeat",
        "kit": "standard kit",
        "fx": {
            "reverb": "plate or room, 1.5s",
            "delay": "subtle 1/8th",
        },
    }

    _BASE_PROFILES = {
        "happy": {
            "drum_style": "pop",
            "density": 0.65,
            "tempo_range": (105, 125),
            "feel": "straight",
            "swing": 0.04,
            "groove": "four_on_the_floor",
            "kit": "tight pop kit",
            "fx": {
                "reverb": "bright plate",
                "delay": "1/8 slap",
                "saturation": "gentle",
            },
        },
        "surprise": {
            "drum_style": "edm",
            "density": 0.7,
            "tempo_range": (120, 138),
            "feel": "straight",
            "swing": 0.0,
            "groove": "build_and_drop",
            "kit": "edm festival kit",
            "fx": {
                "reverb": "big room",
                "delay": "1/4 note",
                "riser": "noise+synth",
            },
        },
        "sad": {
            "drum_style": "jazzy",
            "density": 0.5,
            "tempo_range": (70, 90),
            "feel": "swing",
            "swing": 0.1,
            "groove": "laid_back_swing",
            "kit": "warm kit with brushes",
            "fx": {
                "reverb": "room/plate 1.8s",
                "delay": "tape 1/4 dotted",
            },
        },
        "fear": {
            "drum_style": "minimal",
            "density": 0.55,
            "tempo_range": (75, 110),
            "feel": "straight",
            "swing": 0.02,
            "groove": "pulsing_eighths",
            "kit": "tight electronic kit",
            "fx": {
                "reverb": "long tails",
                "delay": "ping-pong 1/8",
                "filter": "hp+lp moves",
            },
        },
        "angry": {
            "drum_style": "rock",
            "density": 0.75,
            "tempo_range": (120, 150),
            "feel": "straight",
            "swing": 0.0,
            "groove": "driving_backbeat",
            "kit": "aggressive rock kit",
            "fx": {
                "reverb": "tight room",
                "saturation": "heavy",
                "compression": "2-4dB GR",
            },
        },
        "disgust": {
            "drum_style": "industrial",
            "density": 0.6,
            "tempo_range": (95, 125),
            "feel": "straight",
            "swing": 0.0,
            "groove": "mechanical",
            "kit": "processed/industrial kit",
            "fx": {
                "reverb": "metallic plate",
                "distortion": "bitcrush+drive",
            },
        },
    }

    _SUB_OVERRIDES = {
        "happy": {
            "contentment": {
                "density": 0.55,
                "tempo_range": (90, 110),
                "swing": 0.02,
                "groove": "laid_back_backbeat",
            },
        },
        "sad": {
            "grief": {
                "drum_style": "brushes",
                "density": 0.4,
                "tempo_range": (60, 82),
                "feel": "swing",
                "swing": 0.14,
                "groove": "ballad_12_8",
                "kit": "brush kit",
                "fx": {"reverb": "intimate room", "delay": "slow tape"},
            },
            "melancholy": {
                "density": 0.45,
                "tempo_range": (68, 88),
                "swing": 0.12,
                "groove": "loose_backbeat",
            },
        },
        "fear": {
            "anxiety": {
                "density": 0.6,
                "tempo_range": (90, 120),
                "groove": "pulsing_eighths",
            },
        },
        "angry": {
            "rage": {
                "drum_style": "metal",
                "density": 0.85,
                "tempo_range": (135, 170),
                "groove": "double_kick",
                "kit": "metal kit",
                "fx": {"saturation": "crushing", "reverb": "tight room"},
            },
            "frustration": {
                "density": 0.65,
                "tempo_range": (110, 135),
            },
        },
        "disgust": {
            "contempt": {
                "density": 0.55,
                "fx": {"distortion": "bitcrush", "reverb": "short metallic"},
            },
        },
    }

    _GENRE_TO_DRUM_STYLE = {
        "hip-hop": ("hip-hop", 0.12, "boom_bap"),
        "trap": ("trap", 0.08, "trap_rolling"),
        "lofi": ("lofi", 0.16, "lofi_laid_back"),
        "jazz": ("jazzy", 0.14, "swing"),
        "rock": ("rock", 0.02, "driving_backbeat"),
        "metal": ("metal", 0.0, "double_kick"),
        "edm": ("edm", 0.0, "four_on_the_floor"),
        "house": ("edm", 0.0, "four_on_the_floor"),
    }

    _SECTION_DYNAMICS_SHIFT = {
        "intro": -2,
        "verse": -1,
        "pre-chorus": 0,
        "chorus": 1,
        "drop": 2,
        "bridge": -1,
        "outro": -1,
    }

    _SECTION_DENSITY_DELTA = {
        "intro": -0.18,
        "verse": -0.1,
        "pre-chorus": 0.05,
        "chorus": 0.18,
        "drop": 0.2,
        "bridge": -0.05,
        "outro": -0.2,
    }

    _DYNAMIC_STEPS = {
        1: "pp",
        2: "p",
        3: "mp",
        4: "mf",
        5: "f",
        6: "ff",
    }

    def __init__(self, default_genre: Optional[str] = None):
        self.default_genre = default_genre

    def get_production_preset(
        self,
        emotion: EmotionMatch,
        genre: Optional[str] = None,
    ) -> ProductionPreset:
        """
        Return a preset with drum style, dynamics, density, feel, and
        transitions.
        """
        genre_hint = genre or self.default_genre
        profile = self._resolve_profile(emotion)
        drum_style = self.get_drum_style(emotion, genre_hint)
        dynamics_level = self.get_dynamics_level(emotion)
        arrangement_density = self.get_arrangement_density(emotion)
        swing = self._get_swing(profile, genre_hint)
        genre_info = (
            self._GENRE_TO_DRUM_STYLE.get(genre_hint.lower())
            if genre_hint
            else None
        )
        groove_val: str = (
            genre_info[2] if genre_info else None
        ) or str(profile.get("groove", "backbeat"))
        tempo_range = self.get_tempo_range(emotion)
        section_dynamics = self._build_section_dynamics(dynamics_level)
        section_density = self._build_section_density(arrangement_density)

        kit_hint = profile.get("kit", self._NEUTRAL_PROFILE["kit"])
        fx = self._merge_dicts(
            self._NEUTRAL_PROFILE["fx"],
            profile.get("fx", {}),
        )
        transitions = self._build_transition_notes(
            drum_style,
            swing,
            str(profile.get("feel", "straight")),
        )

        notes = {
            "base_emotion": emotion.base_emotion,
            "sub_emotion": emotion.sub_emotion,
            "genre_hint": genre_hint or "unspecified",
            "groove": groove_val,
            "kit": str(kit_hint),
        }

        return ProductionPreset(
            drum_style=drum_style,
            dynamics_level=dynamics_level,
            arrangement_density=arrangement_density,
            intensity_tier=emotion.intensity_tier,
            tempo_range=tempo_range,
            feel="swing" if swing > 0.05 or profile.get(
                "feel") == "swing" else "straight",
            swing=swing,
            groove_motif=groove_val,
            kit_hint=str(kit_hint),
            section_dynamics=section_dynamics,
            section_density=section_density,
            fx=fx,
            transitions=transitions,
            notes=notes,
        )

    def get_drum_style(
        self,
        emotion: EmotionMatch,
        genre: Optional[str] = None,
    ) -> str:
        """
        Choose a drum style seed.

        Genre hint can override emotion defaults (e.g., "hip-hop" forces that).
        """
        if genre:
            style = self._GENRE_TO_DRUM_STYLE.get(genre.lower())
            if style:
                return style[0]

        profile = self._resolve_profile(emotion)
        return profile.get("drum_style", self._NEUTRAL_PROFILE["drum_style"])

    def get_dynamics_level(
        self,
        emotion: EmotionMatch,
        section: Optional[str] = None,
    ) -> str:
        """
        Map intensity tier to a classical dynamic marking.

        Section hints (verse/chorus/bridge) can adjust the level.
        """
        base = self._DYNAMIC_STEPS.get(emotion.intensity_tier, "mf")
        if not section:
            return base

        return self._section_adjusted_dynamic(base, section)

    def get_arrangement_density(self, emotion: EmotionMatch) -> float:
        """
        Convert intensity tier into a 0–1 density suggestion.

        Higher tiers → denser arrangements, with gentle scaling to avoid
        clipping at extremes.
        """
        profile = self._resolve_profile(emotion)
        tier = emotion.intensity_tier or 3
        base_density = float(
            profile.get("density", self._NEUTRAL_PROFILE["density"])
        )
        density = base_density + 0.07 * (tier - 3)
        return self._clamp(density, 0.2, 1.0)

    def get_tempo_range(self, emotion: EmotionMatch) -> Tuple[int, int]:
        """Return a (min, max) tempo range adjusted by intensity tier."""
        profile = self._resolve_profile(emotion)
        tempo_range = cast(
            Tuple[int, int],
            profile.get("tempo_range", self._NEUTRAL_PROFILE["tempo_range"]),
        )
        tier = emotion.intensity_tier or 3
        return self._scale_tempo_range(tempo_range, tier)

    # --- Internal helpers -------------------------------------------------
    def _resolve_profile(self, emotion: EmotionMatch) -> Dict[str, Any]:
        base_key = (emotion.base_emotion or "").lower()
        sub_key = (emotion.sub_emotion or "").lower()

        profile: Dict[str, Any] = self._BASE_PROFILES.get(
            base_key, self._NEUTRAL_PROFILE
        ).copy()
        overrides = self._SUB_OVERRIDES.get(base_key, {}).get(sub_key)
        if overrides:
            profile.update(overrides)
        return profile

    def _get_swing(
        self,
        profile: Dict[str, Any],
        genre: Optional[str],
    ) -> float:
        swing = float(profile.get("swing", 0.0) or 0.0)
        if genre:
            genre_info = self._GENRE_TO_DRUM_STYLE.get(genre.lower())
            if genre_info:
                swing = max(swing, genre_info[1])
        return self._clamp(swing, 0.0, 0.25)

    def _build_section_dynamics(self, base_level: str) -> Dict[str, str]:
        levels = ["pp", "p", "mp", "mf", "f", "ff", "fff"]
        base_idx = (
            levels.index(base_level)
            if base_level in levels
            else levels.index("mf")
        )
        result = {}
        for section, shift in self._SECTION_DYNAMICS_SHIFT.items():
            idx = self._clamp_index(base_idx + shift, len(levels))
            result[section] = levels[idx]
        return result

    def _section_adjusted_dynamic(self, base_level: str, section: str) -> str:
        levels = ["pp", "p", "mp", "mf", "f", "ff", "fff"]
        base_idx = (
            levels.index(base_level)
            if base_level in levels
            else levels.index("mf")
        )
        shift = self._SECTION_DYNAMICS_SHIFT.get(section.lower(), 0)
        idx = self._clamp_index(base_idx + shift, len(levels))
        return levels[idx]

    def _build_section_density(self, base_density: float) -> Dict[str, float]:
        result = {}
        for section, delta in self._SECTION_DENSITY_DELTA.items():
            result[section] = self._clamp(base_density + delta, 0.1, 1.0)
        return result

    def _build_transition_notes(
        self,
        drum_style: str,
        swing: float,
        feel: str,
    ) -> Dict[str, str]:
        fills = (
            "Brush/tom swell fill that preserves swing pocket"
            if swing > 0.05 or feel == "swing"
            else "1-bar tom/snare fill with open hat lift"
        )
        return {
            "into_chorus": f"{fills}; add rise in automation",
            "into_bridge": "Halftime + pull back layers; filter FX down",
            "outro": "Strip to rhythm section and let reverb tails breathe",
        }

    def _scale_tempo_range(
        self,
        tempo_range: Tuple[int, int],
        tier: int,
    ) -> Tuple[int, int]:
        low, high = tempo_range
        shift = (tier - 3) * 3
        return (
            int(self._clamp(low + shift, 40, 220)),
            int(self._clamp(high + shift, 40, 220)),
        )

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def _clamp_index(idx: int, length: int) -> int:
        return max(0, min(length - 1, idx))

    @staticmethod
    def _merge_dicts(
        base: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(base)
        merged.update(overrides)
        return merged
