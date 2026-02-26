"""
MIDI Binding v0 — Symbolic only. Map branch index → fixed MIDI structure.

LISTENING_GUARDRAIL:
MIDI binding is symbolic only.
It must not derive pitch, timing, velocity, or density from intent,
value states, escalation, or constraints.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Tuple

# Fixed mappings. Hard-coded. Not derived from intent or value states.
# Branch A → C major, B → D minor, C → F major
BRANCH_MIDI_NOTES: Tuple[Tuple[int, ...], ...] = (
    (60, 64, 67),  # A: C major root (C4, E4, G4)
    (62, 65, 69),  # B: D minor root
    (65, 69, 72),  # C: F major root
)

# Alternate fixed mapping when pitch_variation token is present. First inversions.
BRANCH_MIDI_NOTES_VARIANT: Tuple[Tuple[int, ...], ...] = (
    (64, 67, 72),  # A: C/E first inv (E4, G4, C5)
    (65, 69, 74),  # B: Dm/F first inv
    (69, 72, 77),  # C: F/A first inv
)

FIXED_VELOCITY = 80
FIXED_DURATION_MS = 400
FIXED_DURATION_MS_VARIANT = 600  # When timing_variation token on. Symbolic only.
EXPECTED_BRANCH_COUNT = 3


def get_midi_for_branch(
    branch_index: int,
    pitch_variation: bool = False,
    timing_variation: bool = False,
) -> List[Tuple[int, int, int]]:
    """
    Return fixed MIDI structure for branch. (pitch, velocity, duration_ms).
    pitch_variation: when True, use alternate fixed mapping (first inversions).
    timing_variation: when True, use alternate fixed duration (600ms).
    Symbolic only. No intent interpretation. All mappings hard-coded.
    """
    if branch_index < 0 or branch_index >= len(BRANCH_MIDI_NOTES):
        branch_index = 0
    mapping = BRANCH_MIDI_NOTES_VARIANT if pitch_variation else BRANCH_MIDI_NOTES
    notes = mapping[branch_index]
    dur = FIXED_DURATION_MS_VARIANT if timing_variation else FIXED_DURATION_MS
    return [(p, FIXED_VELOCITY, dur) for p in notes]


def write_branch_midi_to_file(
    branch_index: int,
    path: Path,
    pitch_variation: bool = False,
    timing_variation: bool = False,
) -> bool:
    """
    Write fixed MIDI structure to .mid file. Symbolic only.
    Returns True on success.
    """
    events = get_midi_for_branch(branch_index, pitch_variation, timing_variation)
    try:
        _write_smf(path, events)
        return True
    except Exception:
        return False


def _write_smf(path: Path, events: List[Tuple[int, int, int]]) -> None:
    """Write minimal Type 0 MIDI file. 120 BPM, 480 ticks/quarter."""
    ticks_per_beat = 480
    # Duration from first event (all same in chord). 120 BPM: 2 beats/sec.
    duration_ms = events[0][2] if events else FIXED_DURATION_MS
    duration_ticks = int(ticks_per_beat * duration_ms / 500)

    def varlen(n: int) -> bytes:
        out = []
        n &= 0x7FFFFFFF
        out.append(n & 0x7F)
        while n > 0x7F:
            n >>= 7
            out.append((n & 0x7F) | 0x80)
        return bytes(reversed(out))

    track_data = bytearray()
    # Note-on all pitches at delta 0
    for pitch, vel, _ in events:
        track_data.extend(varlen(0))
        track_data.extend(bytes([0x90, pitch & 0x7F, vel & 0x7F]))
    # Note-off all pitches after duration (first delta = duration, rest = 0)
    for i, (pitch, _, _) in enumerate(events):
        track_data.extend(varlen(duration_ticks if i == 0 else 0))
        track_data.extend(bytes([0x80, pitch & 0x7F, 0]))
    # End of track
    track_data.extend(varlen(0))
    track_data.extend(bytes([0xFF, 0x2F, 0x00]))

    header = struct.pack(">4sLHHH", b"MThd", 6, 0, 1, ticks_per_beat)
    track_header = struct.pack(">4sL", b"MTrk", len(track_data))
    with open(path, "wb") as f:
        f.write(header)
        f.write(track_header)
        f.write(track_data)
