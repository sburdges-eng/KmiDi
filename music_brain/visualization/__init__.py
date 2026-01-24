"""
Visualization module for music brain.

Provides tools for visualizing musical and emotional data.
"""

from .emotion_trajectory import (
    EmotionSnapshot,
    EmotionTrajectory,
    EmotionTrajectoryVisualizer,
)
from .spectocloud import (
    Anchor,
    AnchorFamily,
    AnchorLibrary,
    EmotionParticle,
    Frame,
    LODConfig,
    LODLevel,
    MusicalParameterExtractor,
    # Performance and LOD (v2.0)
    PerformanceMetrics,
    Spectocloud,
    SpectocloudRenderer,
    StormState,
    # Texturization (v2.0)
    TextureConfig,
    TextureGenerator,
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
    # Performance and LOD (v2.0)
    "PerformanceMetrics",
    "LODLevel",
    "LODConfig",
    # Texturization (v2.0)
    "TextureConfig",
    "TextureGenerator",
]
