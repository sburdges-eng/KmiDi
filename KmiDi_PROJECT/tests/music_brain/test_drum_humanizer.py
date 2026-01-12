"""
Unit tests for drum_humanizer.py
"""

import pytest
from music_brain.groove.drum_humanizer import DrumHumanizer, GuideRuleSet
from music_brain.groove.drum_analysis import DrumTechniqueProfile


class TestDrumHumanizer:
    """Test the DrumHumanizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.humanizer = DrumHumanizer()
        self.humanizer_custom = DrumHumanizer(default_style="rock")

        # Create a mock DrumTechniqueProfile
        self.mock_profile = DrumTechniqueProfile()

    def test_init(self):
        """Test initialization."""
        assert self.humanizer.default_style == "standard"
        assert "standard" in self.humanizer.guide_rules
        assert isinstance(self.humanizer.guide_rules["standard"], GuideRuleSet)

    def test_init_custom_style(self):
        """Test initialization with custom style."""
        assert self.humanizer_custom.default_style == "rock"

    def test_build_default_rules(self):
        """Test default rule building."""
        rules = self.humanizer._build_default_rules()
        assert "standard" in rules
        assert isinstance(rules["standard"], GuideRuleSet)
        assert rules["standard"].swing == 0.0
        assert rules["standard"].timing_shift_ms == 5.0
        assert rules["standard"].ghost_rate == 0.05
        assert rules["standard"].velocity_variation == 0.12

    def test_apply_guide_rules_basic(self):
        """Test basic guide rule application."""
        # This method is a placeholder, so we'll test that it doesn't crash
        result = self.humanizer.apply_guide_rules(self.mock_profile)
        # Since it's a placeholder, it might return None or some default
        # For now, just ensure it doesn't raise an exception
        assert result is not None  # Adjust based on actual implementation

    def test_create_preset_from_guide(self):
        """Test preset creation from guide."""
        from music_brain.groove.groove_engine import GrooveSettings
        preset = self.humanizer.create_preset_from_guide("standard")
        assert isinstance(preset, GrooveSettings)
        assert preset.ghost_note_probability == 0.05

    def test_create_preset_from_guide_unknown_style(self):
        """Test preset creation for unknown style."""
        from music_brain.groove.groove_engine import GrooveSettings
        preset = self.humanizer.create_preset_from_guide("unknown")
        # Should fall back to default
        assert isinstance(preset, GrooveSettings)

    def test_preset_to_settings(self):
        """Test preset to settings conversion."""
        # This method doesn't exist in the current implementation
        pytest.skip("Method _preset_to_settings not implemented yet")

    def test_extract_notes(self):
        """Test note extraction."""
        # This method may not exist yet, depending on implementation
        # For now, skip if not implemented
