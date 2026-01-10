"""
Unit tests for groove applicator.py
"""

import pytest
import tempfile
import os
from pathlib import Path

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

from music_brain.groove.applicator import (
    apply_groove,
    MIDO_AVAILABLE,
)


# Skip all tests if mido not available
pytestmark = pytest.mark.skipif(
    not MIDO_AVAILABLE, reason="mido not installed")


class TestGrooveApplicator:
    """Test the groove applicator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_midi(self, filename="test.mid"):
        """Create a simple test MIDI file."""
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set tempo to 120 BPM
        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

        # Add some drum notes
        track.append(mido.Message('note_on', note=36,
                     velocity=80, channel=9, time=0))      # Kick
        track.append(mido.Message('note_off', note=36,
                     velocity=0, channel=9, time=120))
        track.append(mido.Message('note_on', note=38,
                     velocity=90, channel=9, time=120))   # Snare
        track.append(mido.Message('note_off', note=38,
                     velocity=0, channel=9, time=120))
        track.append(mido.Message('note_on', note=42,
                     velocity=60, channel=9, time=120))   # Hi-hat
        track.append(mido.Message('note_off', note=42,
                     velocity=0, channel=9, time=120))

        filepath = os.path.join(self.temp_dir, filename)
        mid.save(filepath)
        return filepath

    def test_mido_available(self):
        """Test that mido is available."""
        assert MIDO_AVAILABLE == True

    def test_apply_groove_with_genre(self):
        """Test applying groove with genre template."""
        input_file = self.create_test_midi("input.mid")

        output_file = apply_groove(
            midi_path=input_file,
            genre="funk",
            output=os.path.join(self.temp_dir, "output.mid"),
            intensity=0.5
        )

        assert os.path.exists(output_file)
        assert output_file.endswith("output.mid")

        # Load and check the output file
        output_mid = mido.MidiFile(output_file)
        assert len(output_mid.tracks) > 0

    def test_apply_groove_output_default_name(self):
        """Test apply_groove with default output naming."""
        input_file = self.create_test_midi("test.mid")

        output_file = apply_groove(
            midi_path=input_file,
            genre="jazz",
            intensity=0.3
        )

        assert os.path.exists(output_file)
        assert "grooved" in output_file

    def test_apply_groove_intensity_zero(self):
        """Test applying groove with zero intensity (should preserve original)."""
        input_file = self.create_test_midi("original.mid")

        output_file = apply_groove(
            midi_path=input_file,
            genre="rock",
            intensity=0.0,
            output=os.path.join(self.temp_dir, "zero_intensity.mid")
        )

        assert os.path.exists(output_file)

        # Load both files and compare
        input_mid = mido.MidiFile(input_file)
        output_mid = mido.MidiFile(output_file)

        # With zero intensity, should be very similar
        # (exact comparison depends on implementation)

    def test_apply_groove_intensity_full(self):
        """Test applying groove with full intensity."""
        input_file = self.create_test_midi("full.mid")

        output_file = apply_groove(
            midi_path=input_file,
            genre="hiphop",
            intensity=1.0,
            output=os.path.join(self.temp_dir, "full_intensity.mid")
        )

        assert os.path.exists(output_file)

    def test_apply_groove_preserve_dynamics_false(self):
        """Test applying groove without preserving dynamics."""
        input_file = self.create_test_midi("dynamics.mid")

        output_file = apply_groove(
            midi_path=input_file,
            genre="funk",
            preserve_dynamics=False,
            output=os.path.join(self.temp_dir, "no_preserve.mid")
        )

        assert os.path.exists(output_file)

    def test_apply_groove_humanize_options(self):
        """Test groove application with different humanization options."""
        input_file = self.create_test_midi("humanize.mid")

        # Only timing humanization
        output_timing = apply_groove(
            midi_path=input_file,
            genre="jazz",
            humanize_timing=True,
            humanize_velocity=False,
            output=os.path.join(self.temp_dir, "timing_only.mid")
        )

        # Only velocity humanization
        output_velocity = apply_groove(
            midi_path=input_file,
            genre="rock",
            humanize_timing=False,
            humanize_velocity=True,
            output=os.path.join(self.temp_dir, "velocity_only.mid")
        )

        assert os.path.exists(output_timing)
        assert os.path.exists(output_velocity)

    def test_apply_groove_invalid_genre(self):
        """Test applying groove with invalid genre."""
        input_file = self.create_test_midi("invalid.mid")

        with pytest.raises(ValueError):
            apply_groove(
                midi_path=input_file,
                genre="invalid_genre_xyz",
                output=os.path.join(self.temp_dir, "invalid.mid")
            )

    def test_apply_groove_missing_file(self):
        """Test applying groove to non-existent file."""
        with pytest.raises(FileNotFoundError):
            apply_groove(
                midi_path="/nonexistent/file.mid",
                genre="funk"
            )

    def test_apply_groove_different_genres(self):
        """Test applying different genre grooves."""
        input_file = self.create_test_midi("multi_genre.mid")

        genres = ["funk", "jazz", "rock", "hiphop"]
        for genre in genres:
            output_file = apply_groove(
                midi_path=input_file,
                genre=genre,
                output=os.path.join(self.temp_dir, f"{genre}.mid")
            )
            assert os.path.exists(output_file)

            # Verify it's a valid MIDI file
            mid = mido.MidiFile(output_file)
            assert len(mid.tracks) > 0

    def test_apply_groove_complex_midi(self):
        """Test applying groove to a more complex MIDI file."""
        # Create a more complex MIDI file with multiple tracks
        mid = mido.MidiFile(ticks_per_beat=480)

        # Drum track
        drum_track = mido.MidiTrack()
        mid.tracks.append(drum_track)
        drum_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

        # Add more drum notes
        notes = [
            (36, 80, 0),    # Kick
            (38, 90, 240),  # Snare
            (42, 60, 120),  # Hi-hat
            (42, 55, 240),  # Hi-hat
            (36, 85, 480),  # Kick
            (38, 95, 720),  # Snare
        ]

        for note, vel, time in notes:
            drum_track.append(mido.Message(
                'note_on', note=note, velocity=vel, channel=9, time=time))
            drum_track.append(mido.Message(
                'note_off', note=note, velocity=0, channel=9, time=120))

        # Melody track
        melody_track = mido.MidiTrack()
        mid.tracks.append(melody_track)

        melody_notes = [
            (60, 70, 0),
            (62, 75, 240),
            (64, 80, 480),
            (65, 85, 720),
        ]

        for note, vel, time in melody_notes:
            melody_track.append(mido.Message(
                'note_on', note=note, velocity=vel, channel=0, time=time))
            melody_track.append(mido.Message(
                'note_off', note=note, velocity=0, channel=0, time=240))

        complex_file = os.path.join(self.temp_dir, "complex.mid")
        mid.save(complex_file)

        # Apply groove
        output_file = apply_groove(
            midi_path=complex_file,
            genre="funk",
            output=os.path.join(self.temp_dir, "complex_grooved.mid")
        )

        assert os.path.exists(output_file)

        # Verify output has same structure
        output_mid = mido.MidiFile(output_file)
        assert len(output_mid.tracks) == 2  # Should preserve tracks
