"""
Tier1 audio generator wrapper.
"""

from __future__ import annotations

from typing import Dict, Any
from pathlib import Path

from music_brain.audio import render_midi_to_audio


class AudioGenerator:
    def render_from_midi(self, midi_path: str, output_wav: str | None = None) -> Dict[str, Any]:
        midi = Path(midi_path)
        out = output_wav or str(midi.with_suffix(".wav"))
        render_midi_to_audio(str(midi), out)
        return {"midi_path": str(midi), "audio_path": out}
