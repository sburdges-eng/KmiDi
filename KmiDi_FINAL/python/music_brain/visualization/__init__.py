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
    # Performance and LOD (v2.0)
    PerformanceMetrics,
    LODLevel,
    LODConfig,
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
