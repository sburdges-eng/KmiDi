"""
KmiDi JEPA (Joint-Embedding Predictive Architecture) models.

Provides Audio-JEPA, Chord-JEPA, and Stem-JEPA implementations
for self-supervised music understanding.
"""

from music_brain.jepa.audio_jepa import AudioJEPAEncoder, LatentPredictor
from music_brain.jepa.chord_jepa import ChordJEPA
from music_brain.jepa.masking import mask_latents, MultiBlockMasking
from music_brain.jepa.config import (
    JEPAConfig,
    AudioJEPAConfig,
    ChordJEPAConfig,
    StemJEPAConfig,
    TrainingConfig,
)
from music_brain.jepa.datasets import AudioMelDataset, ChordSequenceDataset

__all__ = [
    "AudioJEPAEncoder",
    "LatentPredictor",
    "ChordJEPA",
    "mask_latents",
    "MultiBlockMasking",
    "JEPAConfig",
    "AudioJEPAConfig",
    "ChordJEPAConfig",
    "StemJEPAConfig",
    "TrainingConfig",
    "AudioMelDataset",
    "ChordSequenceDataset",
]
