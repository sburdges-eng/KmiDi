"""
Tier 1: Pretrained models for MIDI/Audio/Voice generation.

No fine-tuning required. Load checkpoints and run inference.
Optimized for Mac (MPS) + CPU/CUDA.

Usage:
    from music_brain.tier1 import Tier1MIDIGenerator, Tier1AudioGenerator

    midi_gen = Tier1MIDIGenerator(device="mps")
    audio_gen = Tier1AudioGenerator(device="mps")
"""

from .midi_generator import Tier1MIDIGenerator
from .audio_generator import Tier1AudioGenerator
from .voice_generator import Tier1VoiceGenerator
from .midi_pipeline import MidiGenerationPipeline, create_midi_pipeline
from .voice_pipeline import (
    VoiceGenerationPipeline,
    VoiceIdentity,
    VocalNote,
    VocalExpression,
    create_voice_pipeline,
)

__all__ = [
    "Tier1MIDIGenerator",
    "Tier1AudioGenerator",
    "Tier1VoiceGenerator",
    "MidiGenerationPipeline",
    "create_midi_pipeline",
    "VoiceGenerationPipeline",
    "VoiceIdentity",
    "VocalNote",
    "VocalExpression",
    "create_voice_pipeline",
]

__version__ = "1.0.0"
