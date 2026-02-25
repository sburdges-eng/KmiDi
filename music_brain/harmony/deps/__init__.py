"""
Harmony dependency bridge.

These imports expose the restored harmony dependencies under:
`music_brain.harmony.deps`.
"""

from music_brain.kelly_companion.utils.harmony_deps.chord_detector import (
    ChordDetector,
    PolyphonicScorer,
    JazzChordAnalyzer,
    ChordMatch,
    ChordQuality,
)
from music_brain.kelly_companion.utils.harmony_deps.key_analyzer import (
    KeyAnalyzer,
    KeyScaleAnalyzer,
    KeyEstimate,
    Mode,
    ModalInterchangeDetector,
    SecondaryDominantDetector,
)
from music_brain.kelly_companion.utils.harmony_deps.harmony_engine import (
    ProbabilisticHarmonyEngine,
    Genre,
    CadenceDetector,
    VoiceLeadingAnalyzer,
    TensionResolutionAnalyzer,
    ChordPrediction,
)
from music_brain.kelly_companion.utils.harmony_deps.chord_memory import (
    ChordMemorySystem,
    SectionType,
    EmotionalArc,
    ChordEvent,
    Phrase,
    PhraseMemory,
    MotifTracker,
)

__all__ = [
    "ChordDetector",
    "PolyphonicScorer",
    "JazzChordAnalyzer",
    "ChordMatch",
    "ChordQuality",
    "KeyAnalyzer",
    "KeyScaleAnalyzer",
    "KeyEstimate",
    "Mode",
    "ModalInterchangeDetector",
    "SecondaryDominantDetector",
    "ProbabilisticHarmonyEngine",
    "Genre",
    "CadenceDetector",
    "VoiceLeadingAnalyzer",
    "TensionResolutionAnalyzer",
    "ChordPrediction",
    "ChordMemorySystem",
    "SectionType",
    "EmotionalArc",
    "ChordEvent",
    "Phrase",
    "PhraseMemory",
    "MotifTracker",
]
