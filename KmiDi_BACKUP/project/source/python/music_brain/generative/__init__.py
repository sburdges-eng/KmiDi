"""
Generative Models for Music Creation.

This module provides a unified interface to various generative models:
- AudioDiffusion: Diffusion-based audio generation
- ChordProgressionGenerator: Transformer-based chord sequence generation
- EmotionConditionedGenerator: Emotion-driven audio/MIDI generation
- MelodyVAE: Variational autoencoder for melody generation
- ArrangementGenerator: Full song arrangement from seeds

All models are designed to run on Apple Silicon (M4 MPS) or CUDA.
"""

from .arrangement import ArrangementGenerator
from .audio_diffusion import AudioDiffusion
from .base import GenerativeConfig, GenerativeModel
from .chord_generator import ChordProgressionGenerator
from .emotion_conditioned import EmotionConditionedGenerator
from .melody_vae import MelodyVAE
from .music_utils import (
    CHROMATIC_NOTES,
    SCALE_INTERVALS,
    get_chord_root,
    get_scale_notes,
    is_minor_key,
    midi_to_note,
    note_to_midi,
    transpose_note,
)

__all__ = [
    "AudioDiffusion",
    "ChordProgressionGenerator",
    "EmotionConditionedGenerator",
    "MelodyVAE",
    "ArrangementGenerator",
    "GenerativeModel",
    "GenerativeConfig",
    # Utilities
    "note_to_midi",
    "midi_to_note",
    "get_scale_notes",
    "transpose_note",
    "is_minor_key",
    "get_chord_root",
    "CHROMATIC_NOTES",
    "SCALE_INTERVALS",
]
