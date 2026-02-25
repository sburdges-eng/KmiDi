"""
Tier1 MIDI generator.
"""

from __future__ import annotations

from typing import Dict, Any, List


class MIDIGenerator:
    """Generate a basic MIDI note sequence from intent-like data."""

    def __init__(self, default_bpm: int = 82):
        self.default_bpm = default_bpm

    def generate(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        melody = intent.get("melody") or [60, 62, 64, 65, 67, 65, 64, 62]
        bpm = int(intent.get("bpm") or self.default_bpm)
        return {"melody": list(melody), "bpm": bpm, "length_bars": 8}
