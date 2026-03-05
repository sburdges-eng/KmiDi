"""
Instrument synthesis bridge from MIDI to rendered audio.
"""

from __future__ import annotations

from music_brain.audio.render import render_midi_to_audio


def synthesize_instruments_from_midi(
    midi_path: str,
    output_wav_path: str,
    sample_rate: int = 44100,
) -> str:
    return render_midi_to_audio(
        midi_path=midi_path, output_path=output_wav_path, sample_rate=sample_rate)
