"""
Harmony Dependencies Package

Provides chord detection, key analysis, harmony generation, and chord memory.
"""

from .chord_detector import (
    ChordDetector,
    PolyphonicScorer,
    JazzChordAnalyzer,
    ChordMatch,
    ChordQuality,
)  # noqa: E501

from .key_analyzer import (
    KeyAnalyzer,
    KeyScaleAnalyzer,
    KeyEstimate,
    Mode,
    ModalInterchangeDetector,
    SecondaryDominantDetector,
)  # noqa: E501

from .harmony_engine import (
    ProbabilisticHarmonyEngine,
    Genre,
    CadenceDetector,
    VoiceLeadingAnalyzer,
    TensionResolutionAnalyzer,
    ChordPrediction,
)  # noqa: E501

from .chord_memory import (
    ChordMemorySystem,
    SectionType,
    EmotionalArc,
    ChordEvent,
    Phrase,
    PhraseMemory,
    MotifTracker,
)  # noqa: E501

# Export main classes with aliases for convenience
HarmonyEngine = ProbabilisticHarmonyEngine

__all__ = [
    # Chord Detection
    "ChordDetector",
    "PolyphonicScorer",
    "JazzChordAnalyzer",
    "ChordMatch",
    "ChordQuality",
    # Key Analysis
    "KeyAnalyzer",
    "KeyScaleAnalyzer",
    "KeyEstimate",
    "Mode",
    "ModalInterchangeDetector",
    "SecondaryDominantDetector",
    # Harmony Engine
    "ProbabilisticHarmonyEngine",
    "HarmonyEngine",  # Alias
    "Genre",
    "CadenceDetector",
    "VoiceLeadingAnalyzer",
    "TensionResolutionAnalyzer",
    "ChordPrediction",
    # Chord Memory
    "ChordMemorySystem",
    "SectionType",
    "EmotionalArc",
    "ChordEvent",
    "Phrase",
    "PhraseMemory",
    "MotifTracker",
]
