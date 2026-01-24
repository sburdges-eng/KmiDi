"""
Penta Core Harmony Package

Advanced harmony implementations including:
- Counterpoint generation
- Jazz chord voicings
- Neo-Riemannian theory
- Microtonal harmony
- Tension analysis
"""

from .counterpoint import *
from .jazz_voicings import *
from .neo_riemannian import *
from .microtonal import *
from .tension import *

__all__ = [
    # Counterpoint
    "CounterpointGenerator",
    "VoiceLeading",
    # Jazz Voicings
    "JazzVoicingGenerator",
    "VoicingStyle",
    # Neo-Riemannian
    "NeoRiemannianTransform",
    "TransformType",
    # Microtonal
    "MicrotonalHarmony",
    "TuningSystem",
    # Tension
    "TensionAnalyzer",
    "TensionProfile",
]
