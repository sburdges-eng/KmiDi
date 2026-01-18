"""
music_brain.tier1 - MIDI Generation Pipeline

This module provides the tier1 MIDI generation components.
It re-exports from the main music_brain package in KmiDi_PROJECT/source/python.
"""

import sys
from pathlib import Path

# Add KmiDi_PROJECT/source/python to path if not already present
_project_python_path = Path(__file__).parent.parent.parent / "KmiDi_PROJECT" / "source" / "python"
if _project_python_path.exists() and str(_project_python_path) not in sys.path:
    sys.path.insert(0, str(_project_python_path))

# Now import from the actual tier1 module
try:
    from music_brain.tier1.midi_pipeline_wrapper import MIDIGenerationPipeline
    from music_brain.tier1.midi_pipeline import MIDIPipeline
    from music_brain.tier1.midi_generator import MIDIGenerator
    from music_brain.tier1.audio_generator import AudioGenerator
    from music_brain.tier1.voice_generator import VoiceGenerator
    from music_brain.tier1.voice_pipeline import VoicePipeline
except ImportError as e:
    # Fallback: define stub classes if imports fail
    import logging
    logging.warning(f"Could not import tier1 modules from KmiDi_PROJECT: {e}")

    class MIDIGenerationPipeline:
        """Stub MIDIGenerationPipeline when actual module is unavailable."""
        def __init__(self, seed=None):
            self.seed = seed

        def generate_midi(self, intent, output_dir="./"):
            return {"status": "stub", "error": "MIDIGenerationPipeline not available"}

__all__ = [
    "MIDIGenerationPipeline",
    "MIDIPipeline",
    "MIDIGenerator",
    "AudioGenerator",
    "VoiceGenerator",
    "VoicePipeline",
]
