"""
Tier1 MIDI pipeline.
"""

from __future__ import annotations

from typing import Dict, Any
from music_brain.tier1.midi_generator import MIDIGenerator


class MIDIPipeline:
    def __init__(self):
        self.generator = MIDIGenerator()

    def run(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        return self.generator.generate(intent)
