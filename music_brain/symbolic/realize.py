"""Symbolic chord-to-MIDI realizer.

A small, library-agnostic chord parser plus a progression-to-MIDI-events
realizer. Covers the common chord qualities a generation pipeline needs
to land notes on the wire: major / minor triads, dom7 / minor7 / maj7
sevenths. Stdlib only — no music21, no mido.

Inputs are textual chord symbols (``"C"``, ``"Am"``, ``"G7"``, ``"Bb"``,
``"F#m"``, ``"Cmaj7"``). Output for the progression realizer is a list
of :class:`NoteEvent` ``(time_sec, pitch, velocity, duration_sec)`` —
the same shape as :class:`music_brain.audio.dual_clip.MidiEvent`, so
the two compose naturally for hybrid symbolic+audio outputs.

Goal-list ties:
  - Symbolic realization layer (direct)
  - Harmonic progression planning (the realization step beneath the
    planner)
  - Hybrid symbolic/audio generation (host-side note emitter)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Pitch-class lookup for chord roots.
_ROOT_PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

# Interval patterns relative to root.
_QUALITY_INTERVALS = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "dom7": (0, 4, 7, 10),
    "minor7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
}

# Order matters: longest suffix first so "maj7" wins over "m7" → "minor7".
_QUALITY_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("maj7", "maj7"),
    ("m7", "minor7"),
    ("7", "dom7"),
    ("m", "minor"),
    ("", "major"),
)

_ROOT_RE = re.compile(r"^([A-G][#b]?)(.*)$")


@dataclass(frozen=True)
class Chord:
    """Parsed chord representation."""

    root_pc: int  # 0-11 (C = 0)
    quality: str
    intervals: Tuple[int, ...]


@dataclass(frozen=True)
class NoteEvent:
    """One sounded note. Same shape as ``MidiEvent`` in audio.dual_clip."""

    time_sec: float
    pitch: int
    velocity: int
    duration_sec: float


def parse_chord(symbol: str) -> Chord:
    """Parse a chord symbol like ``"Am7"`` into a :class:`Chord`.

    Raises :class:`ValueError` on empty input or unrecognised root.
    """
    if not symbol:
        raise ValueError("cannot parse empty chord symbol")
    m = _ROOT_RE.match(symbol)
    if not m:
        raise ValueError(f"unrecognised root in chord symbol: {symbol!r}")
    root_str, suffix = m.group(1), m.group(2)
    if root_str not in _ROOT_PITCH_CLASS:
        raise ValueError(f"unrecognised root note: {root_str!r}")
    root_pc = _ROOT_PITCH_CLASS[root_str]
    quality = "major"
    for suf, qual in _QUALITY_SUFFIXES:
        if suffix == suf:
            quality = qual
            break
    else:
        # `else` on for-else only runs if no break — quality stays 'major'.
        pass
    return Chord(
        root_pc=root_pc,
        quality=quality,
        intervals=_QUALITY_INTERVALS[quality],
    )


def chord_notes(symbol: str, octave: int = 4) -> List[int]:
    """Return MIDI pitch numbers for the chord at the given octave.

    The root is placed at ``octave`` (e.g. ``octave=4`` → C4 = MIDI 60).
    Higher chord tones may wrap into the next octave.
    """
    chord = parse_chord(symbol)
    root_midi = 12 * (octave + 1) + chord.root_pc
    return [root_midi + iv for iv in chord.intervals]


def realize_progression(
    progression: Sequence[dict],
    bpm: float,
    velocity: int,
    octave: int = 4,
) -> List[NoteEvent]:
    """Realize a chord progression as a list of :class:`NoteEvent`.

    Each entry of ``progression`` must be a dict with keys ``"chord"``
    (chord symbol) and ``"duration_beats"`` (length of the chord in
    beats). ``bpm`` converts beats to seconds; ``velocity`` is applied
    uniformly to every emitted note.
    """
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    sec_per_beat = 60.0 / bpm
    out: List[NoteEvent] = []
    cursor_sec = 0.0
    for entry in progression:
        symbol = entry["chord"]
        duration_beats = float(entry["duration_beats"])
        duration_sec = duration_beats * sec_per_beat
        for pitch in chord_notes(symbol, octave=octave):
            out.append(
                NoteEvent(
                    time_sec=cursor_sec,
                    pitch=pitch,
                    velocity=velocity,
                    duration_sec=duration_sec,
                )
            )
        cursor_sec += duration_sec
    return out
