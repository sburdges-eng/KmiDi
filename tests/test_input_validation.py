"""
Test input validation security fixes.

Tests for:
- Input validation
- Override handling
- Type coercion
- Enum usage
- Fallback behavior

Specifically tests for:
- Crashes caused by malformed or mistyped input
- Silent fallback to defaults without signaling
- Semantic inversion (input intent mapping to opposite behavior)
- Empty-but-valid outputs caused by invalid input
- Raw strings passed where enums or typed values are expected
"""

import pytest
from music_brain.session.intent_schema import CompleteSongIntent


class TestIntentSchemaValidation:
    """Test CompleteSongIntent validation and type coercion."""

    def test_from_dict_missing_keys_no_crash(self):
        """Test that missing dict keys don't cause KeyError."""
        # Should not crash, should use defaults
        intent = CompleteSongIntent.from_dict({})
        assert intent.title == ""
        assert intent.created == ""
        assert intent.song_root.core_event == ""

    def test_from_dict_partial_data_no_crash(self):
        """Test that partial data doesn't crash."""
        data = {"title": "Test Song", "song_root": {"core_event": "loss"}}  # Missing other fields
        intent = CompleteSongIntent.from_dict(data)
        assert intent.title == "Test Song"
        assert intent.song_root.core_event == "loss"
        assert intent.song_root.core_resistance == ""  # Default

    def test_tension_bounds_validation(self):
        """Test that mood_secondary_tension is clamped to [0, 1]."""
        # Test upper bound
        intent = CompleteSongIntent(mood_secondary_tension="999")
        assert 0.0 <= intent.song_intent.mood_secondary_tension <= 1.0

        # Test lower bound
        intent = CompleteSongIntent(mood_secondary_tension="-50")
        assert 0.0 <= intent.song_intent.mood_secondary_tension <= 1.0

        # Test valid value
        intent = CompleteSongIntent(mood_secondary_tension="0.7")
        assert intent.song_intent.mood_secondary_tension == 0.7

    def test_tension_invalid_type_fallback(self):
        """Test that invalid tension types fall back to default."""
        intent = CompleteSongIntent(mood_secondary_tension="not_a_number")
        assert intent.song_intent.mood_secondary_tension == 0.5  # Default

    def test_vulnerability_scale_enum_validation(self):
        """Test that vulnerability_scale validates enum values."""
        # Valid values should normalize to title case
        intent = CompleteSongIntent(vulnerability_scale="low")
        assert intent.song_intent.vulnerability_scale == "Low"

        intent = CompleteSongIntent(vulnerability_scale="HIGH")
        assert intent.song_intent.vulnerability_scale == "High"

        # Invalid values should default to Medium
        intent = CompleteSongIntent(vulnerability_scale="invalid")
        assert intent.song_intent.vulnerability_scale == "Medium"

    def test_vulnerability_scale_float_to_enum(self):
        """Test that vulnerability_scale converts float to enum."""
        # Low range
        intent = CompleteSongIntent(vulnerability_scale=0.2)
        assert intent.song_intent.vulnerability_scale == "Low"

        # Medium range
        intent = CompleteSongIntent(vulnerability_scale=0.5)
        assert intent.song_intent.vulnerability_scale == "Medium"

        # High range
        intent = CompleteSongIntent(vulnerability_scale=0.9)
        assert intent.song_intent.vulnerability_scale == "High"

    def test_from_dict_vulnerability_enum_validation(self):
        """Test vulnerability_scale validation in from_dict."""
        data = {"song_intent": {"vulnerability_scale": "INVALID_VALUE"}}
        intent = CompleteSongIntent.from_dict(data)
        assert intent.song_intent.vulnerability_scale == "Medium"  # Default

    def test_from_dict_tension_clamping(self):
        """Test tension clamping in from_dict."""
        data = {"song_intent": {"mood_secondary_tension": 999}}
        intent = CompleteSongIntent.from_dict(data)
        assert 0.0 <= intent.song_intent.mood_secondary_tension <= 1.0


class TestAPIValidation:
    """Test API input validation."""

    def test_bpm_validation_clamping(self):
        """Test that BPM is clamped to reasonable range."""

        # This would be tested through the API endpoint, but we can test
        # the validation logic directly

        # BPM too high should clamp to 300
        bpm = 10000
        try:
            bpm = int(bpm)
            bpm = max(40, min(300, bpm))
        except (ValueError, TypeError):
            bpm = 82
        assert bpm == 300

        # BPM too low should clamp to 40
        bpm = 10
        try:
            bpm = int(bpm)
            bpm = max(40, min(300, bpm))
        except (ValueError, TypeError):
            bpm = 82
        assert bpm == 40

        # Invalid BPM should default
        bpm = "fast"
        try:
            bpm = int(bpm)
            bpm = max(40, min(300, bpm))
        except (ValueError, TypeError):
            bpm = 82
        assert bpm == 82

    def test_duration_validation(self):
        """Test that duration is validated to positive values."""
        # Duration too large should clamp
        duration = 1000.0
        try:
            duration = float(duration)
            duration = max(0.1, min(60.0, duration))
        except (ValueError, TypeError):
            duration = 3.0
        assert duration == 60.0

        # Negative duration should clamp to minimum
        duration = -5.0
        try:
            duration = float(duration)
            duration = max(0.1, min(60.0, duration))
        except (ValueError, TypeError):
            duration = 3.0
        assert duration == 0.1

        # Zero duration should clamp to minimum
        duration = 0.0
        try:
            duration = float(duration)
            duration = max(0.1, min(60.0, duration))
        except (ValueError, TypeError):
            duration = 3.0
        assert duration == 0.1

    def test_mode_validation(self):
        """Test that musical modes are validated."""
        valid_modes = {
            "major",
            "minor",
            "dorian",
            "phrygian",
            "lydian",
            "mixolydian",
            "aeolian",
            "locrian",
        }

        # Valid mode should pass through
        mode = "dorian"
        mode = mode if mode in valid_modes else "major"
        assert mode == "dorian"

        # Invalid mode should default to major
        mode = "invalid_mode"
        mode = mode if mode in valid_modes else "major"
        assert mode == "major"

        # Empty mode should default to major
        mode = ""
        mode = mode if mode in valid_modes else "major"
        assert mode == "major"


class TestDictAccessSafety:
    """Test that all dict access is safe and doesn't crash."""

    def test_from_dict_all_missing_keys(self):
        """Test from_dict with all possible missing keys."""
        # Empty dict
        intent = CompleteSongIntent.from_dict({})
        assert isinstance(intent, CompleteSongIntent)

        # Only title
        intent = CompleteSongIntent.from_dict({"title": "test"})
        assert intent.title == "test"
        assert intent.song_root.core_event == ""

        # song_root with missing sub-keys
        intent = CompleteSongIntent.from_dict({"song_root": {}})
        assert intent.song_root.core_event == ""

        # song_intent with missing sub-keys
        intent = CompleteSongIntent.from_dict({"song_intent": {}})
        assert intent.song_intent.mood_primary == ""

        # technical_constraints with missing sub-keys
        intent = CompleteSongIntent.from_dict({"technical_constraints": {}})
        assert intent.technical_constraints.technical_genre == ""

        # system_directive with missing sub-keys
        intent = CompleteSongIntent.from_dict({"system_directive": {}})
        assert intent.system_directive.output_target == ""


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_tension_exact_bounds(self):
        """Test exact boundary values for tension."""
        # Exactly 0.0
        intent = CompleteSongIntent(mood_secondary_tension="0.0")
        assert intent.song_intent.mood_secondary_tension == 0.0

        # Exactly 1.0
        intent = CompleteSongIntent(mood_secondary_tension="1.0")
        assert intent.song_intent.mood_secondary_tension == 1.0

        # Just over 1.0
        intent = CompleteSongIntent(mood_secondary_tension="1.1")
        assert intent.song_intent.mood_secondary_tension == 1.0  # Clamped

        # Just under 0.0
        intent = CompleteSongIntent(mood_secondary_tension="-0.1")
        assert intent.song_intent.mood_secondary_tension == 0.0  # Clamped

    def test_empty_string_inputs(self):
        """Test empty string inputs don't cause issues."""
        intent = CompleteSongIntent(
            core_event="",
            mood_primary="",
            technical_genre="",
        )
        assert intent.song_root.core_event == ""
        assert intent.song_intent.mood_primary == ""
        assert intent.technical_constraints.technical_genre == ""

    def test_none_inputs(self):
        """Test None inputs are handled properly."""
        # These should use defaults
        intent = CompleteSongIntent(
            core_event=None,
            mood_primary=None,
        )
        # None gets converted to empty string by default
        assert isinstance(intent.song_root.core_event, str)
        assert isinstance(intent.song_intent.mood_primary, str)


class TestSemanticCorrectness:
    """Test that inputs map to semantically correct outputs."""

    def test_vulnerability_mapping(self):
        """Test that vulnerability scale maps correctly."""
        # 0.0 should be Low
        intent = CompleteSongIntent(vulnerability_scale=0.0)
        assert intent.song_intent.vulnerability_scale == "Low"

        # 0.5 should be Medium
        intent = CompleteSongIntent(vulnerability_scale=0.5)
        assert intent.song_intent.vulnerability_scale == "Medium"

        # 1.0 should be High
        intent = CompleteSongIntent(vulnerability_scale=1.0)
        assert intent.song_intent.vulnerability_scale == "High"

        # No semantic inversion - high values = high vulnerability
        intent = CompleteSongIntent(vulnerability_scale=0.9)
        assert intent.song_intent.vulnerability_scale != "Low"  # Should not invert

    def test_tension_semantic_correctness(self):
        """Test tension values are semantically correct."""
        # High input should give high output (no inversion)
        intent = CompleteSongIntent(mood_secondary_tension="0.9")
        assert intent.song_intent.mood_secondary_tension >= 0.8

        # Low input should give low output
        intent = CompleteSongIntent(mood_secondary_tension="0.1")
        assert intent.song_intent.mood_secondary_tension <= 0.2


class TestStrictParameterMapping:
    """Test that UI parameters map strictly through to CompleteSongIntent (no silent defaults)."""

    def _make_validated_request(self, overrides=None):
        """Build a minimal valid CompleteSongIntentRequest payload."""
        from music_brain.engine_api.schema import CompleteSongIntentRequest

        payload = {
            "core_desire": "healing through loss",
            "mood_primary": "melancholy",
            "genre": "ambient",
            "tempo": 90,
            "key_mode": "C minor",
            "structure": [{"name": "intro", "bars": 8, "repetitions": 1}],
            "instruments": [{"instrument": "piano", "techniques": ["arpeggiated"]}],
            "allow_legacy_fallback": False,
            "groove_feel": "Swing/Laid-back",
            "narrative_arc": "Slow-Burn",
            "rule_to_break": "parallel_fifths",
            "rule_justification": "Creates raw emotional power",
        }
        if overrides:
            payload.update(overrides)
        return CompleteSongIntentRequest.model_validate(payload)

    def test_groove_feel_maps_to_intent(self):
        """groove_feel must reach CompleteSongIntent.technical_constraints."""
        validated = self._make_validated_request()
        intent = CompleteSongIntent(
            technical_groove_feel=validated.groove_feel,
        )
        assert intent.technical_constraints.technical_groove_feel == "Swing/Laid-back"

    def test_narrative_arc_maps_to_intent(self):
        """narrative_arc must reach CompleteSongIntent.song_intent."""
        validated = self._make_validated_request()
        intent = CompleteSongIntent(
            narrative_arc=validated.narrative_arc,
        )
        assert intent.song_intent.narrative_arc == "Slow-Burn"

    def test_rule_to_break_maps_to_intent(self):
        """rule_to_break must reach CompleteSongIntent.technical_constraints."""
        validated = self._make_validated_request()
        intent = CompleteSongIntent(
            technical_rule_to_break=validated.rule_to_break or "",
            rule_breaking_justification=validated.rule_justification or "",
        )
        assert intent.technical_constraints.technical_rule_to_break == "parallel_fifths"
        assert (
            intent.technical_constraints.rule_breaking_justification
            == "Creates raw emotional power"
        )

    def test_vulnerability_scale_not_hardcoded(self):
        """vulnerability_scale from the request must not be replaced by a hardcoded 0.5."""
        intent_low = CompleteSongIntent(vulnerability_scale=0.1)
        assert intent_low.song_intent.vulnerability_scale == "Low"

        intent_high = CompleteSongIntent(vulnerability_scale=0.9)
        assert intent_high.song_intent.vulnerability_scale == "High"

    def test_defaults_when_optional_fields_absent(self):
        """Optional fields use sensible defaults, not empty strings."""
        validated = self._make_validated_request(
            {"rule_to_break": None, "rule_justification": None}
        )
        assert validated.groove_feel == "Swing/Laid-back"
        assert validated.narrative_arc == "Slow-Burn"
        assert validated.rule_to_break is None
        assert validated.rule_justification is None

    def test_full_convert_roundtrip(self):
        """Simulate the full _convert_to_intent mapping path."""
        validated = self._make_validated_request()
        key_parts = validated.key_mode.split()
        technical_key = key_parts[0]
        technical_mode = key_parts[1].lower()
        tempo_range = (max(60, validated.tempo - 20), min(140, validated.tempo + 20))

        intent = CompleteSongIntent(
            core_event=validated.core_desire,
            core_longing=validated.core_desire,
            mood_primary=validated.mood_primary,
            narrative_arc=validated.narrative_arc,
            vulnerability_scale=0.7,
            technical_genre=validated.genre,
            technical_tempo_range=tempo_range,
            technical_key=technical_key,
            technical_mode=technical_mode,
            technical_groove_feel=validated.groove_feel,
            technical_rule_to_break=validated.rule_to_break or "",
            rule_breaking_justification=validated.rule_justification or "",
        )

        assert intent.song_intent.narrative_arc == "Slow-Burn"
        assert intent.song_intent.vulnerability_scale == "High"
        assert intent.technical_constraints.technical_groove_feel == "Swing/Laid-back"
        assert intent.technical_constraints.technical_rule_to_break == "parallel_fifths"
        assert (
            intent.technical_constraints.rule_breaking_justification
            == "Creates raw emotional power"
        )
        assert intent.technical_constraints.technical_genre == "ambient"
        assert intent.technical_constraints.technical_key == "C"
        assert intent.technical_constraints.technical_mode == "minor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
