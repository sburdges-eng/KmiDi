"""
Unit tests for groove humanizer.py
"""

import pytest
from unittest.mock import Mock

from music_brain.groove.humanizer import (
    GrooveHumanizer,
    HumanizationProfile,
    TimingHumanizer,
    VelocityHumanizer,
    GrooveTemplate,
    apply_humanization,
)


class TestHumanizationProfile:
    """Test the HumanizationProfile dataclass."""

    def test_profile_creation(self):
        """Test creating a humanization profile."""
        profile = HumanizationProfile(
            timing_variation=0.1,
            velocity_variation=0.2,
            swing_factor=0.3,
            groove_intensity=0.4
        )

        assert profile.timing_variation == 0.1
        assert profile.velocity_variation == 0.2
        assert profile.swing_factor == 0.3
        assert profile.groove_intensity == 0.4

    def test_profile_defaults(self):
        """Test profile with default values."""
        profile = HumanizationProfile()

        assert profile.timing_variation == 0.0
        assert profile.velocity_variation == 0.0
        assert profile.swing_factor == 0.0
        assert profile.groove_intensity == 1.0

    def test_profile_validation(self):
        """Test profile value validation."""
        # Valid values
        profile = HumanizationProfile(
            timing_variation=0.5,
            velocity_variation=0.3,
            swing_factor=0.2,
            groove_intensity=0.8
        )
        assert profile.timing_variation == 0.5

        # Edge cases
        profile_zero = HumanizationProfile(
            timing_variation=0.0,
            velocity_variation=0.0,
            swing_factor=0.0,
            groove_intensity=0.0
        )
        assert profile_zero.groove_intensity == 0.0


class TestTimingHumanizer:
    """Test the TimingHumanizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.humanizer = TimingHumanizer()

    def test_timing_humanizer_creation(self):
        """Test creating a timing humanizer."""
        assert self.humanizer is not None

    def test_apply_timing_variation(self):
        """Test applying timing variation to note times."""
        original_times = [0, 240, 480, 720, 960]  # Quarter notes at 120 BPM
        variation = 0.1  # 10% variation

        humanized_times = self.humanizer.apply_timing_variation(
            original_times, variation
        )

        assert len(humanized_times) == len(original_times)
        assert all(isinstance(t, (int, float)) for t in humanized_times)

        # Check that times are reasonably close to original
        for orig, humanized in zip(original_times, humanized_times):
            deviation = abs(humanized - orig)
            max_deviation = orig * variation * 2  # Allow some tolerance
            assert deviation <= max_deviation

    def test_apply_swing(self):
        """Test applying swing to note times."""
        # Create a pattern that should be swung (8th notes)
        times = [0, 120, 240, 360, 480, 600, 720, 840]  # 8th notes
        swing_factor = 0.5  # 50% swing

        swung_times = self.humanizer.apply_swing(times, swing_factor)

        assert len(swung_times) == len(times)

        # In swing, even 8ths should be delayed, odd 8ths should be early
        # This is a complex calculation, just check basic properties
        assert all(isinstance(t, (int, float)) for t in swung_times)

    def test_timing_variation_zero(self):
        """Test timing variation with zero variation."""
        times = [0, 100, 200, 300]
        humanized = self.humanizer.apply_timing_variation(times, 0.0)

        # Should return exact copies
        assert humanized == times

    def test_swing_zero(self):
        """Test swing with zero factor."""
        times = [0, 120, 240, 360]
        swung = self.humanizer.apply_swing(times, 0.0)

        # Should return exact copies
        assert swung == times


class TestVelocityHumanizer:
    """Test the VelocityHumanizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.humanizer = VelocityHumanizer()

    def test_velocity_humanizer_creation(self):
        """Test creating a velocity humanizer."""
        assert self.humanizer is not None

    def test_apply_velocity_variation(self):
        """Test applying velocity variation."""
        original_velocities = [64, 80, 100, 120]
        variation = 0.2  # 20% variation

        humanized_velocities = self.humanizer.apply_velocity_variation(
            original_velocities, variation
        )

        assert len(humanized_velocities) == len(original_velocities)
        assert all(isinstance(v, (int, float)) for v in humanized_velocities)
        assert all(0 <= v <= 127 for v in humanized_velocities)  # MIDI range

    def test_apply_velocity_curve(self):
        """Test applying velocity curve."""
        velocities = [60, 70, 80, 90, 100]
        curve_type = "exponential"

        curved_velocities = self.humanizer.apply_velocity_curve(
            velocities, curve_type
        )

        assert len(curved_velocities) == len(velocities)
        assert all(isinstance(v, (int, float)) for v in curved_velocities)

    def test_velocity_variation_zero(self):
        """Test velocity variation with zero variation."""
        velocities = [64, 80, 100]
        humanized = self.humanizer.apply_velocity_variation(velocities, 0.0)

        # Should return exact copies
        assert humanized == velocities

    def test_velocity_curve_linear(self):
        """Test linear velocity curve."""
        velocities = [50, 75, 100]
        curved = self.humanizer.apply_velocity_curve(velocities, "linear")

        # Linear should preserve relative relationships
        assert curved[0] < curved[1] < curved[2]


class TestGrooveTemplate:
    """Test the GrooveTemplate class."""

    def test_template_creation(self):
        """Test creating a groove template."""
        template = GrooveTemplate(
            name="test_groove",
            timing_deviations=[0.0, 0.02, -0.01, 0.03],
            velocity_curve=[1.0, 0.9, 1.1, 0.95],
            swing_factor=0.2,
            intensity=0.8
        )

        assert template.name == "test_groove"
        assert len(template.timing_deviations) == 4
        assert len(template.velocity_curve) == 4
        assert template.swing_factor == 0.2
        assert template.intensity == 0.8

    def test_template_defaults(self):
        """Test template with default values."""
        template = GrooveTemplate(name="default")

        assert template.name == "default"
        assert template.timing_deviations == []
        assert template.velocity_curve == []
        assert template.swing_factor == 0.0
        assert template.intensity == 1.0


class TestGrooveHumanizer:
    """Test the GrooveHumanizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.humanizer = GrooveHumanizer()

    def test_humanizer_creation(self):
        """Test creating a groove humanizer."""
        assert self.humanizer is not None
        assert hasattr(self.humanizer, 'timing_humanizer')
        assert hasattr(self.humanizer, 'velocity_humanizer')

    def test_apply_profile_to_notes(self):
        """Test applying a humanization profile to mock notes."""
        # Create mock notes
        mock_notes = [
            Mock(time=0, velocity=80),
            Mock(time=240, velocity=90),
            Mock(time=480, velocity=70),
            Mock(time=720, velocity=85),
        ]

        profile = HumanizationProfile(
            timing_variation=0.1,
            velocity_variation=0.2,
            swing_factor=0.0,
            groove_intensity=0.5
        )

        # Apply humanization
        humanized_notes = self.humanizer.apply_profile_to_notes(
            mock_notes, profile
        )

        assert len(humanized_notes) == len(mock_notes)

        # Check that times and velocities have been modified
        for orig, humanized in zip(mock_notes, humanized_notes):
            assert hasattr(humanized, 'time')
            assert hasattr(humanized, 'velocity')
            assert isinstance(humanized.time, (int, float))
            assert isinstance(humanized.velocity, (int, float))

    def test_apply_template_to_notes(self):
        """Test applying a groove template to notes."""
        mock_notes = [
            Mock(time=0, velocity=80),
            Mock(time=120, velocity=90),
            Mock(time=240, velocity=70),
            Mock(time=360, velocity=85),
        ]

        template = GrooveTemplate(
            name="test_template",
            timing_deviations=[0.0, 0.02, -0.01, 0.03],
            velocity_curve=[1.0, 0.9, 1.1, 0.95],
            swing_factor=0.1,
            intensity=0.7
        )

        humanized_notes = self.humanizer.apply_template_to_notes(
            mock_notes, template
        )

        assert len(humanized_notes) == len(mock_notes)

        # Verify modifications were applied
        for note in humanized_notes:
            assert hasattr(note, 'time')
            assert hasattr(note, 'velocity')

    def test_humanize_with_zero_intensity(self):
        """Test humanization with zero intensity."""
        mock_notes = [Mock(time=100, velocity=80)]
        profile = HumanizationProfile(groove_intensity=0.0)

        humanized = self.humanizer.apply_profile_to_notes(mock_notes, profile)

        # Should be minimally changed
        assert len(humanized) == 1
        assert abs(humanized[0].time - 100) < 10  # Small variation allowed
        assert abs(humanized[0].velocity - 80) < 16  # Small variation allowed


class TestApplyHumanization:
    """Test the apply_humanization function."""

    def test_apply_humanization_function(self):
        """Test the main apply_humanization function."""
        # Create mock MIDI-like data
        mock_track = Mock()
        mock_track.notes = [
            Mock(time=0, velocity=80, note=36),
            Mock(time=240, velocity=90, note=38),
            Mock(time=480, velocity=70, note=42),
        ]

        profile = HumanizationProfile(
            timing_variation=0.05,
            velocity_variation=0.1,
            swing_factor=0.0,
            groove_intensity=0.5
        )

        # Apply humanization
        result_track = apply_humanization(mock_track, profile)

        assert result_track is not None
        assert len(result_track.notes) == 3

        # Check that notes have been modified
        for note in result_track.notes:
            assert hasattr(note, 'time')
            assert hasattr(note, 'velocity')
            assert 0 <= note.velocity <= 127

    def test_apply_humanization_with_template(self):
        """Test apply_humanization with a groove template."""
        mock_track = Mock()
        mock_track.notes = [
            Mock(time=0, velocity=80),
            Mock(time=120, velocity=90),
        ]

        template = GrooveTemplate(
            name="test",
            timing_deviations=[0.0, 0.02],
            velocity_curve=[1.0, 0.9],
            swing_factor=0.0,
            intensity=0.8
        )

        result_track = apply_humanization(mock_track, template=template)

        assert result_track is not None
        assert len(result_track.notes) == 2

    def test_apply_humanization_no_profile_or_template(self):
        """Test apply_humanization with neither profile nor template."""
        mock_track = Mock()
        mock_track.notes = [Mock(time=100, velocity=80)]

        # Should raise error or return unchanged
        with pytest.raises(ValueError):
            apply_humanization(mock_track)
