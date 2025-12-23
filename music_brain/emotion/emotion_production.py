"""
Emotion → production mapper.

Provides lightweight, code-first defaults that mirror the Production_Workflows
guides (drums, dynamics, arrangement) so other modules can request a
ProductionPreset from an EmotionMatch without needing to touch the guides
directly. This is an initial scaffold; enrich with genre- and guide-specific
rules as integration matures.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from music_brain.emotion_thesaurus import EmotionMatch


@dataclass
class ProductionPreset:
    """Container for production decisions derived from an emotion."""

    drum_style: str = "standard"          # e.g., rock, hip-hop, jazzy
    dynamics_level: str = "mf"            # pp → fff
    arrangement_density: float = 0.5      # 0–1, sparse to dense
    intensity_tier: Optional[int] = None  # 1–6 from EmotionMatch
    notes: Dict[str, str] = field(default_factory=dict)


class EmotionProductionMapper:
    """
    Map EmotionMatch → ProductionPreset.

    The defaults lean on common genre conventions:
    - happy/surprise → brighter, denser, pop/edm-friendly drums
    - sad/fear → sparser, laid-back, acoustic/jazzy drums
    - angry/disgust → aggressive, tight, rock/industrial drums
    """

    _BASE_DRUM_STYLE = {
        "happy": "pop",
        "surprise": "edm",
        "sad": "jazzy",
        "fear": "minimal",
        "angry": "rock",
        "disgust": "industrial",
    }

    _DYNAMIC_STEPS = {
        1: "mp",
        2: "mp",
        3: "mf",
        4: "f",
        5: "ff",
        6: "fff",
    }

    def __init__(self, default_genre: Optional[str] = None):
        self.default_genre = default_genre

    def get_production_preset(
        self,
        emotion: EmotionMatch,
        genre: Optional[str] = None,
    ) -> ProductionPreset:
        """Return a preset with drum style, dynamics, and density hints."""
        genre_hint = genre or self.default_genre
        return ProductionPreset(
            drum_style=self.get_drum_style(emotion, genre_hint),
            dynamics_level=self.get_dynamics_level(emotion),
            arrangement_density=self.get_arrangement_density(emotion),
            intensity_tier=emotion.intensity_tier,
            notes={
                "base_emotion": emotion.base_emotion,
                "sub_emotion": emotion.sub_emotion,
                "genre_hint": genre_hint or "unspecified",
            },
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
            return genre.lower()
        return self._BASE_DRUM_STYLE.get(emotion.base_emotion.lower(), "standard")

    def get_dynamics_level(
        self,
        emotion: EmotionMatch,
        section: Optional[str] = None,
    ) -> str:
        """
        Map intensity tier to a classical dynamic marking.

        Section hints (verse/chorus/bridge) can be layered later by the
        DynamicsEngine; for now we return a single baseline.
        """
        return self._DYNAMIC_STEPS.get(emotion.intensity_tier, "mf")

    def get_arrangement_density(self, emotion: EmotionMatch) -> float:
        """
        Convert intensity tier into a 0–1 density suggestion.

        Higher tiers → denser arrangements, with gentle scaling to avoid
        clipping at extremes.
        """
        tier = emotion.intensity_tier or 3
        base = 0.35 + 0.1 * max(1, min(tier, 6))
        return max(0.2, min(1.0, base))

