"""
Audio rendering helpers for MIDI outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import wave
import math
import struct


def _note_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _parse_midi_notes(midi_path: Path) -> List[Dict[str, float]]:
    try:
        import mido  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mido is required for MIDI-to-audio rendering") from exc

    mid = mido.MidiFile(str(midi_path))
    tempo = 500000
    active: Dict[int, float] = {}
    notes: List[Dict[str, float]] = []
    now = 0.0
    for msg in mid:
        now += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
            active[msg.note] = now
            continue
        if msg.type in ("note_off", "note_on"):
            start = active.pop(msg.note, None)
            if start is not None and now > start:
                notes.append({"pitch": float(msg.note), "start": start, "end": now})
    return notes


def render_midi_to_audio(
    midi_path: str,
    output_path: str,
    sample_rate: int = 44100,
) -> str:
    """
    Render a MIDI file to a simple mono WAV.

    Uses sine-wave synthesis as a dependency-light fallback renderer.
    """
    midi_file = Path(midi_path)
    if not midi_file.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    notes = _parse_midi_notes(midi_file)
    total_duration = max((n["end"] for n in notes), default=1.0) + 0.2
    total_samples = max(1, int(total_duration * sample_rate))
    pcm = [0.0] * total_samples

    for note in notes:
        freq = _note_freq(int(note["pitch"]))
        start_idx = max(0, int(note["start"] * sample_rate))
        end_idx = min(total_samples, int(note["end"] * sample_rate))
        attack = int(0.01 * sample_rate)
        release = int(0.03 * sample_rate)
        for i in range(start_idx, end_idx):
            t = (i - start_idx) / sample_rate
            env = 1.0
            if i - start_idx < attack:
                env = max(0.0, (i - start_idx) / max(1, attack))
            if end_idx - i < release:
                env *= max(0.0, (end_idx - i) / max(1, release))
            pcm[i] += 0.18 * env * math.sin(2.0 * math.pi * freq * t)

    # soft clip
    for idx, sample in enumerate(pcm):
        if sample > 1.0:
            pcm[idx] = 1.0
        elif sample < -1.0:
            pcm[idx] = -1.0

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = (struct.pack("<h", int(s * 32767.0)) for s in pcm)
        wf.writeframes(b"".join(frames))
    return str(out)
