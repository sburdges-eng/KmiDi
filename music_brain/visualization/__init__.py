"""
Visualization module for music brain.

Provides tools for visualizing musical and emotional data.
"""

from .emotion_trajectory import (
    EmotionTrajectoryVisualizer,
    EmotionTrajectory,
    EmotionSnapshot,
)

from .spectocloud import (
    Spectocloud,
    Anchor,
    AnchorFamily,
    Frame,
    EmotionParticle,
    StormState,
    MusicalParameterExtractor,
    AnchorLibrary,
    SpectocloudRenderer,
)

__all__ = [
    "EmotionTrajectoryVisualizer",
    "EmotionTrajectory",
    "EmotionSnapshot",
    "Spectocloud",
    "Anchor",
    "AnchorFamily",
    "Frame",
    "EmotionParticle",
    "StormState",
    "MusicalParameterExtractor",
    "AnchorLibrary",
    "SpectocloudRenderer",
]
