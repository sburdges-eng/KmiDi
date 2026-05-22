"""Chord-to-MIDI realizer (Goal: Symbolic realization layer,
Harmonic progression planning, Hybrid symbolic/audio generation)."""

import pytest

from music_brain.symbolic.realize import (
    chord_notes,
    parse_chord,
    realize_progression,
)


# ---------------------------------------------------------------------------
# parse_chord
# ---------------------------------------------------------------------------


def test_parse_major_triad():
    chord = parse_chord("C")
    assert chord.root_pc == 0  # C
    assert chord.quality == "major"
    assert chord.intervals == (0, 4, 7)


def test_parse_minor_triad():
    chord = parse_chord("Am")
    assert chord.root_pc == 9  # A
    assert chord.quality == "minor"
    assert chord.intervals == (0, 3, 7)


def test_parse_seventh_chord():
    chord = parse_chord("G7")
    assert chord.root_pc == 7  # G
    assert chord.quality == "dom7"
    # Dominant 7th: root, M3, P5, m7
    assert chord.intervals == (0, 4, 7, 10)


def test_parse_minor_seventh():
    chord = parse_chord("Dm7")
    assert chord.root_pc == 2  # D
    assert chord.quality == "minor7"
    assert chord.intervals == (0, 3, 7, 10)


def test_parse_major_seventh():
    chord = parse_chord("Cmaj7")
    assert chord.root_pc == 0
    assert chord.quality == "maj7"
    assert chord.intervals == (0, 4, 7, 11)


def test_parse_sharp_root():
    chord = parse_chord("F#m")
    assert chord.root_pc == 6  # F#
    assert chord.quality == "minor"


def test_parse_flat_root():
    chord = parse_chord("Bb")
    assert chord.root_pc == 10  # Bb
    assert chord.quality == "major"


def test_parse_invalid_root_raises():
    with pytest.raises(ValueError, match="root"):
        parse_chord("H7")


def test_parse_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_chord("")


# ---------------------------------------------------------------------------
# chord_notes (MIDI pitch numbers)
# ---------------------------------------------------------------------------


def test_chord_notes_default_octave_is_4():
    """C major triad in octave 4 → MIDI 60 (C4), 64 (E4), 67 (G4)."""
    notes = chord_notes("C")
    assert notes == [60, 64, 67]


def test_chord_notes_custom_octave():
    """Am in octave 3 → A3 (57), C4 (60), E4 (64)."""
    notes = chord_notes("Am", octave=3)
    assert notes == [57, 60, 64]


def test_chord_notes_seventh_chord_has_four_notes():
    notes = chord_notes("G7", octave=4)
    # G4=67, B4=71, D5=74, F5=77
    assert notes == [67, 71, 74, 77]


# ---------------------------------------------------------------------------
# realize_progression
# ---------------------------------------------------------------------------


def test_realize_progression_outputs_one_event_per_chord_note():
    progression = [
        {"chord": "C", "duration_beats": 4},
        {"chord": "Am", "duration_beats": 4},
    ]
    events = realize_progression(progression, bpm=120, velocity=80)
    # 2 chords × 3 notes each = 6 events.
    assert len(events) == 6
    # All same velocity.
    assert all(e.velocity == 80 for e in events)


def test_realize_progression_event_times_advance_with_duration():
    progression = [
        {"chord": "C", "duration_beats": 2},
        {"chord": "F", "duration_beats": 2},
    ]
    events = realize_progression(progression, bpm=120, velocity=80)
    # 2 beats at 120bpm = 1.0 second.
    first_chord = [e for e in events if e.time_sec == pytest.approx(0.0)]
    second_chord = [e for e in events if e.time_sec == pytest.approx(1.0)]
    assert len(first_chord) == 3
    assert len(second_chord) == 3


def test_realize_progression_duration_matches_chord_length():
    progression = [{"chord": "C", "duration_beats": 4}]
    events = realize_progression(progression, bpm=120, velocity=80)
    # 4 beats at 120bpm = 2.0 seconds per note.
    assert all(e.duration_sec == pytest.approx(2.0) for e in events)


def test_realize_progression_empty_returns_empty():
    assert realize_progression([], bpm=120, velocity=80) == []


def test_realize_progression_invalid_bpm_raises():
    with pytest.raises(ValueError, match="bpm"):
        realize_progression([{"chord": "C", "duration_beats": 4}], bpm=0, velocity=80)
