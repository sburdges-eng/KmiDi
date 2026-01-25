"""
Export module for music brain.

Provides enhanced export functionality with emotion metadata and social platform optimization.
"""

from .emotion_stem_exporter import (
    EmotionMetadata,
    EmotionStemExporter,
    StemExportInfo,
    create_emotion_metadata_from_intent,
)
from .social_platform_exporter import (
    PLATFORM_SPECS,
    PlatformSpec,
    SocialPlatform,
    SocialPlatformExporter,
)

__all__ = [
    # Emotion stem export
    "EmotionStemExporter",
    "EmotionMetadata",
    "StemExportInfo",
    "create_emotion_metadata_from_intent",
    # Social platform export
    "SocialPlatformExporter",
    "SocialPlatform",
    "PlatformSpec",
    "PLATFORM_SPECS",
]
