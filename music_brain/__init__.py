"""
music_brain - Music Intelligence Toolkit

This package provides session/intent management and integrations.
For the full music_brain toolkit including tier1/tier2 modules,
ensure KmiDi_PROJECT/source/python is in your PYTHONPATH.
"""

__version__ = "0.2.0"

# Re-export commonly used items
from .session.intent_schema import (
    CompleteSongIntent,
    SongIntent,
    SongRoot,
    TechnicalConstraints,
    SystemDirective,
)

__all__ = [
    "CompleteSongIntent",
    "SongIntent",
    "SongRoot",
    "TechnicalConstraints",
    "SystemDirective",
]
