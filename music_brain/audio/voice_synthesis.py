"""
High-level voice synthesis bridge.
"""

from __future__ import annotations

from typing import Iterable

from music_brain.voice import VoiceSynthesizer, get_voice_profile


def synthesize_guide_vocal(
    lyrics: str,
    melody_midi: Iterable[int],
    output_path: str,
    profile: str = "guide_vulnerable",
    tempo_bpm: int = 82,
) -> str:
    synth = VoiceSynthesizer(get_voice_profile(profile))
    return synth.synthesize_guide(
        lyrics=lyrics,
        melody_midi=melody_midi,
        tempo_bpm=tempo_bpm,
        output_path=output_path,
    )
