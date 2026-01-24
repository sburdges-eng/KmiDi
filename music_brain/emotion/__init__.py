"""
Emotion package.

Re-exports the emotion thesaurus alongside production-mapping helpers.
"""

from music_brain.emotion.emotion_production import (
    EmotionProductionMapper,
    ProductionPreset,
)
from music_brain.emotion.emotion_thesaurus import BlendMatch, EmotionMatch, EmotionThesaurus

__all__ = [
    "EmotionThesaurus",
    "EmotionMatch",
    "BlendMatch",
    "EmotionProductionMapper",
    "ProductionPreset",
]
