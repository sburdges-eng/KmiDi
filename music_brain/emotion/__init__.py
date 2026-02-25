"""
Emotion package.

Re-exports the emotion thesaurus alongside production-mapping helpers
and audio-based emotion classification from trained ML models.
"""

from music_brain.emotion.emotion_thesaurus import EmotionThesaurus, EmotionMatch, BlendMatch
from music_brain.emotion.emotion_production import (
    EmotionProductionMapper,
    ProductionPreset,
)

# Audio emotion classification (lazy import to avoid torch dependency if not needed)
def get_audio_classifier():
    """Get the audio emotion classifier (lazy import)."""
    from music_brain.emotion.audio_emotion_classifier import AudioEmotionClassifier
    return AudioEmotionClassifier

def classify_audio_emotion(audio_path: str):
    """Quick audio emotion classification."""
    from music_brain.emotion.audio_emotion_classifier import classify_audio_emotion as _classify
    return _classify(audio_path)

__all__ = [
    "EmotionThesaurus",
    "EmotionMatch",
    "BlendMatch",
    "EmotionProductionMapper",
    "ProductionPreset",
    "get_audio_classifier",
    "classify_audio_emotion",
]
