"""
Unit tests for emotion_production.py
"""

import pytest
from music_brain.emotion.emotion_production import EmotionProductionMapper, ProductionPreset
from music_brain.emotion.emotion_thesaurus import EmotionMatch


class TestEmotionProductionMapper:
    """Test the EmotionProductionMapper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = EmotionProductionMapper()
        self.mapper_with_genre = EmotionProductionMapper(
            default_genre="hip-hop")

        # Create test EmotionMatch objects
        self.happy_emotion = EmotionMatch(
            base_emotion="happy",
            sub_emotion="joy",
            sub_sub_emotion="ecstatic",
            intensity_tier=4,
            matched_synonym="joyful",
            all_tier_synonyms=["happy", "joyful", "cheerful"],
            emotion_id="HAPPY-JOY-ECSTATIC",
            description="Pure joy and happiness"
        )

        self.sad_emotion = EmotionMatch(
            base_emotion="sad",
            sub_emotion="grief",
            sub_sub_emotion="bereaved",
            intensity_tier=2,
            matched_synonym="grieving",
            all_tier_synonyms=["sad", "grieving", "melancholy"],
            emotion_id="SAD-GRIEF-BEREAVED",
            description="Deep sadness and loss"
        )

    def test_get_production_preset_basic(self):
        """Test basic preset generation."""
        preset = self.mapper.get_production_preset(self.happy_emotion)

        assert isinstance(preset, ProductionPreset)
        assert preset.drum_style == "pop"
        assert preset.dynamics_level == "f"
        assert preset.arrangement_density == 0.75  # 0.35 + 0.1 * 4 = 0.75
        assert preset.intensity_tier == 4
        assert preset.notes["base_emotion"] == "happy"
        assert preset.notes["sub_emotion"] == "joy"
        assert preset.notes["genre_hint"] == "unspecified"

    def test_get_production_preset_with_genre(self):
        """Test preset generation with genre override."""
        preset = self.mapper.get_production_preset(
            self.happy_emotion, genre="rock")

        assert preset.drum_style == "rock"
        assert preset.notes["genre_hint"] == "rock"

    def test_get_production_preset_default_genre(self):
        """Test preset generation with default genre."""
        preset = self.mapper_with_genre.get_production_preset(self.sad_emotion)

        assert preset.drum_style == "hip-hop"
        assert preset.notes["genre_hint"] == "hip-hop"

    def test_get_drum_style_emotion_based(self):
        """Test drum style selection based on emotion."""
        assert self.mapper.get_drum_style(self.happy_emotion) == "pop"
        assert self.mapper.get_drum_style(self.sad_emotion) == "jazzy"

    def test_get_drum_style_genre_override(self):
        """Test drum style with genre override."""
        assert self.mapper.get_drum_style(self.happy_emotion, "edm") == "edm"

    def test_get_drum_style_unknown_emotion(self):
        """Test drum style for unknown emotion."""
        unknown_emotion = EmotionMatch(
            base_emotion="unknown",
            sub_emotion="weird",
            sub_sub_emotion="strange",
            intensity_tier=3,
            matched_synonym="weird",
            all_tier_synonyms=["weird"],
            emotion_id="UNKNOWN-WEIRD-STRANGE",
            description="Unknown emotion"
        )
        assert self.mapper.get_drum_style(unknown_emotion) == "standard"

    def test_get_dynamics_level(self):
        """Test dynamics level mapping."""
        assert self.mapper.get_dynamics_level(
            self.happy_emotion) == "f"  # tier 4
        assert self.mapper.get_dynamics_level(
            self.sad_emotion) == "mp"  # tier 2

    def test_get_dynamics_level_unknown_tier(self):
        """Test dynamics level for unknown tier."""
        emotion_no_tier = EmotionMatch(
            base_emotion="neutral",
            sub_emotion="calm",
            sub_sub_emotion="balanced",
            intensity_tier=None,
            matched_synonym="calm",
            all_tier_synonyms=["calm"],
            emotion_id="NEUTRAL-CALM-BALANCED",
            description="Neutral emotion"
        )
        assert self.mapper.get_dynamics_level(
            emotion_no_tier) == "mf"  # default

    def test_get_arrangement_density(self):
        """Test arrangement density calculation."""
        # Tier 4: 0.35 + 0.1 * 4 = 0.75
        assert self.mapper.get_arrangement_density(self.happy_emotion) == 0.75

        # Tier 2: 0.35 + 0.1 * 2 = 0.55
        assert self.mapper.get_arrangement_density(self.sad_emotion) == 0.55

    def test_get_arrangement_density_edge_cases(self):
        """Test arrangement density edge cases."""
        low_tier = EmotionMatch(
            base_emotion="sad",
            sub_emotion="grief",
            sub_sub_emotion="bereaved",
            intensity_tier=1,
            matched_synonym="sad",
            all_tier_synonyms=["sad"],
            emotion_id="SAD-GRIEF-BEREAVED",
            description="Very sad"
        )
        high_tier = EmotionMatch(
            base_emotion="angry",
            sub_emotion="rage",
            sub_sub_emotion="furious",
            intensity_tier=6,
            matched_synonym="furious",
            all_tier_synonyms=["furious"],
            emotion_id="ANGRY-RAGE-FURIOUS",
            description="Very angry"
        )

        # Tier 1: 0.35 + 0.1 * 1 = 0.45
        assert abs(self.mapper.get_arrangement_density(
            low_tier) - 0.45) < 1e-10

        # Tier 6: 0.35 + 0.1 * 6 = 0.95
        assert abs(self.mapper.get_arrangement_density(
            high_tier) - 0.95) < 1e-10

    def test_intensity_tier_scaling(self):
        """Test that intensity tier affects all mappings."""
        tier1 = EmotionMatch(
            base_emotion="sad",
            sub_emotion="grief",
            sub_sub_emotion="bereaved",
            intensity_tier=1,
            matched_synonym="sad",
            all_tier_synonyms=["sad"],
            emotion_id="SAD-GRIEF-BEREAVED",
            description="Tier 1"
        )
        tier6 = EmotionMatch(
            base_emotion="happy",
            sub_emotion="joy",
            sub_sub_emotion="ecstatic",
            intensity_tier=6,
            matched_synonym="ecstatic",
            all_tier_synonyms=["ecstatic"],
            emotion_id="HAPPY-JOY-ECSTATIC",
            description="Tier 6"
        )

        preset1 = self.mapper.get_production_preset(tier1)
        preset6 = self.mapper.get_production_preset(tier6)

        assert preset1.dynamics_level == "mp"  # tier 1
        assert preset6.dynamics_level == "fff"  # tier 6

        assert preset1.arrangement_density < preset6.arrangement_density

    def test_genre_overrides(self):
        """Test that genre overrides emotion-based defaults."""
        rock_preset = self.mapper.get_production_preset(
            self.happy_emotion, genre="rock")
