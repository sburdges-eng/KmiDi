"""
Production-level processing modules (arrangement, dynamics, automation).
"""

from music_brain.production.dynamics_engine import (
    AutomationCurve,
    DynamicsEngine,
    SectionDynamics,
    SongStructure,
)
from music_brain.production.emotion_production import (
    EmotionProductionMapper,
    ProductionPreset,
)

__all__ = [
    "DynamicsEngine",
    "SectionDynamics",
    "SongStructure",
    "AutomationCurve",
    "EmotionProductionMapper",
    "ProductionPreset",
]
