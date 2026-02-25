"""
Tier1 voice pipeline.
"""

from __future__ import annotations

from typing import Dict, Any
from music_brain.tier1.voice_generator import VoiceGenerator


class VoicePipeline:
    def __init__(self):
        self.generator = VoiceGenerator()

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.generator.generate(
            lyrics=payload.get("lyrics", ""),
            melody_midi=payload.get("melody", [60, 62, 64, 65]),
            output_path=payload.get("output_path", "guide_vocal.wav"),
            tempo_bpm=int(payload.get("tempo_bpm", 82)),
        )
