"""
Cross-cultural music support for KmiDi.

Provides integration for non-Western musical systems:
- Indian Classical (Raga)
- Arabic/Middle Eastern (Maqam)
- East Asian Pentatonic systems (Chinese, Japanese, Korean)
"""

from music_brain.cultural.cross_cultural_music import (
    CrossCulturalMusicMapper,
    RagaSystem,
    MaqamSystem,
    EastAsianPentatonicSystem,
    get_cultural_scale_for_emotion,
)

__all__ = [
    "CrossCulturalMusicMapper",
    "RagaSystem",
    "MaqamSystem",
    "EastAsianPentatonicSystem",
    "get_cultural_scale_for_emotion",
]

