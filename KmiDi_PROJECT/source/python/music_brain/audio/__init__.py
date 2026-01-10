"""
Audio Analysis - Analyze feel and characteristics of audio files.

Features:
- Feel/groove analysis
- Energy extraction
- Tempo detection
- Key detection
- Spectral analysis
- Reference track DNA analysis
- Chord detection from audio
- 8-band frequency analysis
- Comprehensive audio analysis
- Comprehensive audio analysis (AudioAnalyzer)
- Audio refinery (sample processing)
"""

from music_brain.audio.feel import analyze_feel, AudioFeatures
from music_brain.audio.reference_dna import (
    analyze_reference,
    apply_reference_to_plan,
    ReferenceProfile,
)
from music_brain.audio.chord_detection import (
    ChordDetector,
    ChordDetection,
    ChordProgressionDetection,
)
from music_brain.audio.frequency_analysis import (
    analyze_frequency_bands,
    compare_frequency_profiles,
    suggest_eq_adjustments,
    FrequencyProfile,
)
from music_brain.audio.analyzer import (
    AudioAnalyzer,
    AudioAnalysis,
    analyze_audio,
)
from music_brain.audio.refinery import (
    process_file,
    refine_folder,
    run_refinery,
    pipe_clean,
    pipe_industrial,
    pipe_tape_rot,
)

__all__ = [
    "analyze_feel",
    "AudioFeatures",
    # Reference DNA
    "analyze_reference",
    "apply_reference_to_plan",
    "ReferenceProfile",
    # Chord detection
    "ChordDetector",
    "ChordDetection",
    "ChordProgressionDetection",
    # Frequency analysis
    "analyze_frequency_bands",
    "compare_frequency_profiles",
    "suggest_eq_adjustments",
    "FrequencyProfile",
    # Comprehensive analyzer
    "AudioAnalyzer",
    "AudioAnalysis",
    "analyze_audio",
    # Audio refinery
    "process_file",
    "refine_folder",
    "run_refinery",
    "pipe_clean",
    "pipe_industrial",
    "pipe_tape_rot",
]
