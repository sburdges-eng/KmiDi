"""
Production-level processing modules (arrangement, dynamics, automation).
"""

from music_brain.production.dynamics_engine import (
    AutomationCurve,
    DynamicsEngine,
    SectionDynamics,
    SongStructure,
)

__all__ = [
    "DynamicsEngine",
    "SectionDynamics",
    "SongStructure",
    "AutomationCurve",
]
