"""Tests for MIDI generator."""
import pytest
from kelly.core.midi_generator import MidiGenerator, GrooveTemplate


def test_midi_generator_initialization():
    """Test MIDI generator initializes."""
    generator = MidiGenerator(tempo=120)
    assert generator.tempo == 120
    assert len(generator.groove_templates) > 0


def test_groove_templates():
    """Test groove templates are available."""
    generator = MidiGenerator()
    assert "straight" in generator.groove_templates
    assert "swing" in generator.groove_templates
    assert "syncopated" in generator.groove_templates


def test_generate_chord_progression_major():
    """Test generating major chord progression."""
    generator = MidiGenerator()
    progression = generator.generate_chord_progression(mode="major", length=4)
    assert len(progression) == 4
    assert all(isinstance(chord, list) for chord in progression)


def test_generate_chord_progression_minor():
    """Test generating minor chord progression."""
    generator = MidiGenerator()
    progression = generator.generate_chord_progression(mode="minor", length=4)
    assert len(progression) == 4


def test_chord_progression_with_dissonance():
    """Test chord progression with dissonance."""
    generator = MidiGenerator()
    progression = generator.generate_chord_progression(
        mode="minor",
        allow_dissonance=True
    )
    # Dissonant chords should have more notes
    assert any(len(chord) > 3 for chord in progression)


def test_create_midi_file():
    """Test creating MIDI file."""
    generator = MidiGenerator()
    progression = generator.generate_chord_progression(mode="major")
    midi_file = generator.create_midi_file(progression, groove="straight")
    assert midi_file is not None
    assert len(midi_file.tracks) > 0
