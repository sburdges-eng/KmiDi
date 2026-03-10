"""
Tier1 voice generator wrapper.
"""

from __future__ import annotations

from typing import Dict, Any, Iterable

from music_brain.voice import VoiceSynthesizer, get_voice_profile


class VoiceGenerator:
    def __init__(self, profile: str = "guide_vulnerable"):
        self.profile = profile

    def generate(
        self,
        lyrics: str,
        melody_midi: Iterable[int],
        output_path: str = "guide_vocal.wav",
        tempo_bpm: int = 82,
    ) -> Dict[str, Any]:
        synth = VoiceSynthesizer(get_voice_profile(self.profile))
        audio_path = synth.synthesize_guide(
            lyrics, melody_midi, tempo_bpm=tempo_bpm, output_path=output_path)
        return {"audio_path": audio_path, "profile": self.profile}
