"""
Export module for music brain.

Provides enhanced export functionality with emotion metadata and social platform optimization.
"""

from .emotion_stem_exporter import (
    EmotionStemExporter,
    EmotionMetadata,
    StemExportInfo,
    create_emotion_metadata_from_intent,
)

from .social_platform_exporter import (
    SocialPlatformExporter,
    SocialPlatform,
    PlatformSpec,
    PLATFORM_SPECS,
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
