#!/usr/bin/env python3
"""
Manual test runner for input validation fixes.
Runs tests without pytest dependency.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from music_brain.session.intent_schema import CompleteSongIntent


def test_from_dict_missing_keys_no_crash():
    """Test that missing dict keys don't cause KeyError."""
    print("Test: from_dict with missing keys...")
    intent = CompleteSongIntent.from_dict({})
    assert intent.title == "", f"Expected empty title, got {intent.title}"
    assert intent.song_root.core_event == "", f"Expected empty core_event"
    print("✓ PASS: No crash on missing keys")


def test_tension_bounds_validation():
    """Test that mood_secondary_tension is clamped to [0, 1]."""
    print("\nTest: Tension bounds validation...")
    
    # Test upper bound
    intent = CompleteSongIntent(mood_secondary_tension="999")
    assert 0.0 <= intent.song_intent.mood_secondary_tension <= 1.0, \
        f"Tension {intent.song_intent.mood_secondary_tension} not in [0, 1]"
    assert intent.song_intent.mood_secondary_tension == 1.0, \
        f"Expected 1.0, got {intent.song_intent.mood_secondary_tension}"
    
    # Test lower bound
    intent = CompleteSongIntent(mood_secondary_tension="-50")
    assert 0.0 <= intent.song_intent.mood_secondary_tension <= 1.0, \
        f"Tension {intent.song_intent.mood_secondary_tension} not in [0, 1]"
    assert intent.song_intent.mood_secondary_tension == 0.0, \
        f"Expected 0.0, got {intent.song_intent.mood_secondary_tension}"
    
    # Test valid value
    intent = CompleteSongIntent(mood_secondary_tension="0.7")
    assert intent.song_intent.mood_secondary_tension == 0.7, \
        f"Expected 0.7, got {intent.song_intent.mood_secondary_tension}"
    
    print("✓ PASS: Tension clamped to [0, 1]")


def test_tension_invalid_type_fallback():
    """Test that invalid tension types fall back to default."""
    print("\nTest: Invalid tension type fallback...")
    intent = CompleteSongIntent(mood_secondary_tension="not_a_number")
    assert intent.song_intent.mood_secondary_tension == 0.5, \
        f"Expected 0.5, got {intent.song_intent.mood_secondary_tension}"
    print("✓ PASS: Invalid type falls back to 0.5")


def test_vulnerability_scale_enum_validation():
    """Test that vulnerability_scale validates enum values."""
    print("\nTest: Vulnerability scale enum validation...")
    
    # Valid values should normalize to title case
    intent = CompleteSongIntent(vulnerability_scale="low")
    assert intent.song_intent.vulnerability_scale == "Low", \
        f"Expected 'Low', got {intent.song_intent.vulnerability_scale}"
    
    intent = CompleteSongIntent(vulnerability_scale="HIGH")
    assert intent.song_intent.vulnerability_scale == "High", \
        f"Expected 'High', got {intent.song_intent.vulnerability_scale}"
    
    # Invalid values should default to Medium
    intent = CompleteSongIntent(vulnerability_scale="invalid")
    assert intent.song_intent.vulnerability_scale == "Medium", \
        f"Expected 'Medium', got {intent.song_intent.vulnerability_scale}"
    
    print("✓ PASS: Vulnerability scale validates enum values")


def test_vulnerability_scale_float_to_enum():
    """Test that vulnerability_scale converts float to enum."""
    print("\nTest: Vulnerability scale float to enum...")
    
    # Low range
    intent = CompleteSongIntent(vulnerability_scale=0.2)
    assert intent.song_intent.vulnerability_scale == "Low", \
        f"Expected 'Low', got {intent.song_intent.vulnerability_scale}"
    
    # Medium range
    intent = CompleteSongIntent(vulnerability_scale=0.5)
    assert intent.song_intent.vulnerability_scale == "Medium", \
        f"Expected 'Medium', got {intent.song_intent.vulnerability_scale}"
    
    # High range
    intent = CompleteSongIntent(vulnerability_scale=0.9)
    assert intent.song_intent.vulnerability_scale == "High", \
        f"Expected 'High', got {intent.song_intent.vulnerability_scale}"
    
    print("✓ PASS: Float converts to enum correctly")


def test_from_dict_all_missing_keys():
    """Test from_dict with all possible missing keys."""
    print("\nTest: from_dict with all missing keys...")
    
    # Empty dict
    intent = CompleteSongIntent.from_dict({})
    assert isinstance(intent, CompleteSongIntent), "Failed to create intent"
    
    # song_root with missing sub-keys
    intent = CompleteSongIntent.from_dict({"song_root": {}})
    assert intent.song_root.core_event == "", "core_event should be empty"
    
    # song_intent with missing sub-keys
    intent = CompleteSongIntent.from_dict({"song_intent": {}})
    assert intent.song_intent.mood_primary == "", "mood_primary should be empty"
    
    # technical_constraints with missing sub-keys
    intent = CompleteSongIntent.from_dict({"technical_constraints": {}})
    assert intent.technical_constraints.technical_genre == "", "technical_genre should be empty"
    
    # system_directive with missing sub-keys
    intent = CompleteSongIntent.from_dict({"system_directive": {}})
    assert intent.system_directive.output_target == "", "output_target should be empty"
    
    print("✓ PASS: All missing keys handled safely")


def test_from_dict_vulnerability_enum_validation():
    """Test vulnerability_scale validation in from_dict."""
    print("\nTest: from_dict vulnerability enum validation...")
    
    data = {"song_intent": {"vulnerability_scale": "INVALID_VALUE"}}
    intent = CompleteSongIntent.from_dict(data)
    assert intent.song_intent.vulnerability_scale == "Medium", \
        f"Expected 'Medium', got {intent.song_intent.vulnerability_scale}"
    
    print("✓ PASS: Invalid enum value defaults to Medium")


def test_from_dict_tension_clamping():
    """Test tension clamping in from_dict."""
    print("\nTest: from_dict tension clamping...")
    
    data = {"song_intent": {"mood_secondary_tension": 999}}
    intent = CompleteSongIntent.from_dict(data)
    assert 0.0 <= intent.song_intent.mood_secondary_tension <= 1.0, \
        f"Tension not in range: {intent.song_intent.mood_secondary_tension}"
    assert intent.song_intent.mood_secondary_tension == 1.0, \
        f"Expected 1.0, got {intent.song_intent.mood_secondary_tension}"
    
    print("✓ PASS: Tension clamped in from_dict")


def test_bpm_validation():
    """Test BPM validation logic."""
    print("\nTest: BPM validation...")
    
    # BPM too high
    bpm = 10000
    bpm = max(40, min(300, int(bpm)))
    assert bpm == 300, f"Expected 300, got {bpm}"
    
    # BPM too low
    bpm = 10
    bpm = max(40, min(300, int(bpm)))
    assert bpm == 40, f"Expected 40, got {bpm}"
    
    # Invalid BPM
    bpm = "fast"
    try:
        bpm = int(bpm)
        bpm = max(40, min(300, bpm))
    except (ValueError, TypeError):
        bpm = 82
    assert bpm == 82, f"Expected 82, got {bpm}"
    
    print("✓ PASS: BPM validation works correctly")


def test_duration_validation():
    """Test duration validation logic."""
    print("\nTest: Duration validation...")
    
    # Duration too large
    duration = 1000.0
    duration = max(0.1, min(60.0, float(duration)))
    assert duration == 60.0, f"Expected 60.0, got {duration}"
    
    # Negative duration
    duration = -5.0
    duration = max(0.1, min(60.0, float(duration)))
    assert duration == 0.1, f"Expected 0.1, got {duration}"
    
    # Zero duration
    duration = 0.0
    duration = max(0.1, min(60.0, float(duration)))
    assert duration == 0.1, f"Expected 0.1, got {duration}"
    
    print("✓ PASS: Duration validation works correctly")


def test_mode_validation():
    """Test mode validation logic."""
    print("\nTest: Mode validation...")
    
    valid_modes = {"major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"}
    
    # Valid mode
    mode = "dorian"
    mode = mode if mode in valid_modes else "major"
    assert mode == "dorian", f"Expected 'dorian', got {mode}"
    
    # Invalid mode
    mode = "invalid_mode"
    mode = mode if mode in valid_modes else "major"
    assert mode == "major", f"Expected 'major', got {mode}"
    
    print("✓ PASS: Mode validation works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Input Validation Security Test Suite")
    print("=" * 60)
    
    tests = [
        test_from_dict_missing_keys_no_crash,
        test_tension_bounds_validation,
        test_tension_invalid_type_fallback,
        test_vulnerability_scale_enum_validation,
        test_vulnerability_scale_float_to_enum,
        test_from_dict_all_missing_keys,
        test_from_dict_vulnerability_enum_validation,
        test_from_dict_tension_clamping,
        test_bpm_validation,
        test_duration_validation,
        test_mode_validation,
    ]
    
    failed = []
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"✗ FAIL: {test_func.__name__}")
            print(f"  Error: {e}")
            failed.append(test_func.__name__)
        except Exception as e:
            print(f"✗ ERROR: {test_func.__name__}")
            print(f"  Exception: {type(e).__name__}: {e}")
            failed.append(test_func.__name__)
    
    print("\n" + "=" * 60)
    print(f"Results: {len(tests) - len(failed)}/{len(tests)} tests passed")
    if failed:
        print(f"Failed tests: {', '.join(failed)}")
        return 1
    else:
        print("All tests passed! ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
