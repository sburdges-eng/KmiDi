"""
Emotion package.

Re-exports the emotion thesaurus alongside production-mapping helpers.
"""

from music_brain.emotion_thesaurus import EmotionThesaurus, EmotionMatch, BlendMatch
from music_brain.emotion.emotion_production import (
    EmotionProductionMapper,
    ProductionPreset,
)

__all__ = [
    "EmotionThesaurus",
    "EmotionMatch",
    "BlendMatch",
    "EmotionProductionMapper",
    "ProductionPreset",
]

