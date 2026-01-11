#!/usr/bin/env python3
"""
Validation script for cross-cultural music mappings.

Tests that Raga, Maqam, and pentatonic scales can be correctly mapped to emotions
and that the scales are properly structured.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_brain.cultural.cross_cultural_music import (
    CrossCulturalMusicMapper,
    MusicSystem,
    RagaSystem,
    MaqamSystem,
    EastAsianPentatonicSystem,
)


def validate_ragas():
    """Validate Raga system."""
    print("\n=== Validating Raga System ===")
    raga_system = RagaSystem()
    
    test_emotions = ["sad", "happy", "calm", "melancholy", "joy", "peace", "grief", "love"]
    passed = 0
    failed = 0
    
    for emotion in test_emotions:
        raga = raga_system.find_raga_for_emotion(emotion, 0.5)
        if raga:
            raga_data = raga_system.get_raga(raga)
            if raga_data:
                print(f"  ✓ {emotion} → {raga} (culture: {raga_data.culture})")
                passed += 1
            else:
                print(f"  ✗ {emotion} → {raga} (failed to load)")
                failed += 1
        else:
            print(f"  ⚠ {emotion} → No raga found")
            failed += 1
    
    print(f"\nRaga validation: {passed} passed, {failed} failed")
    return failed == 0


def validate_maqamat():
    """Validate Maqam system."""
    print("\n=== Validating Maqam System ===")
    maqam_system = MaqamSystem()
    
    test_emotions = ["sad", "happy", "mystery", "nostalgia", "passion", "drama", "calm"]
    passed = 0
    failed = 0
    
    for emotion in test_emotions:
        maqam = maqam_system.find_maqam_for_emotion(emotion, 0.5)
        if maqam:
            maqam_data = maqam_system.get_maqam(maqam)
            if maqam_data:
                print(f"  ✓ {emotion} → {maqam} (culture: {maqam_data.culture})")
                passed += 1
            else:
                print(f"  ✗ {emotion} → {maqam} (failed to load)")
                failed += 1
        else:
            print(f"  ⚠ {emotion} → No maqam found")
            failed += 1
    
    print(f"\nMaqam validation: {passed} passed, {failed} failed")
    return failed == 0


def validate_pentatonic():
    """Validate pentatonic system."""
    print("\n=== Validating East Asian Pentatonic System ===")
    pentatonic_system = EastAsianPentatonicSystem()
    
    test_emotions = ["peace", "melancholy", "meditation", "cheer", "calm", "grief"]
    cultures = [None, "Chinese", "Japanese", "Korean"]
    passed = 0
    failed = 0
    
    for emotion in test_emotions:
        for culture in cultures:
            scale = pentatonic_system.find_scale_for_emotion(emotion, 0.5, culture)
            if scale:
                scale_data = pentatonic_system.get_scale(scale)
                if scale_data:
                    culture_label = culture or "Any"
                    print(f"  ✓ {emotion} ({culture_label}) → {scale} (culture: {scale_data.culture})")
                    passed += 1
                else:
                    print(f"  ✗ {emotion} ({culture}) → {scale} (failed to load)")
                    failed += 1
                break  # Only show first match per emotion
    
    print(f"\nPentatonic validation: {passed} passed, {failed} failed")
    return failed == 0


def validate_integration():
    """Validate integration with CrossCulturalMusicMapper."""
    print("\n=== Validating Integration ===")
    mapper = CrossCulturalMusicMapper()
    
    test_emotions = ["sad", "happy", "calm", "mystery", "peace"]
    passed = 0
    failed = 0
    
    for emotion in test_emotions:
        all_scales = mapper.get_all_systems_for_emotion(emotion, 0.5)
        found_any = False
        
        for system_name, scale in all_scales.items():
            if scale:
                found_any = True
                print(f"  ✓ {emotion} → {system_name}: {scale.name} (intervals: {scale.intervals_semitones})")
                passed += 1
        
        if not found_any:
            print(f"  ⚠ {emotion} → No scales found in any system")
            failed += 1
    
    print(f"\nIntegration validation: {passed} passed, {failed} failed")
    return failed == 0


def validate_scale_structure():
    """Validate that all scales have proper structure."""
    print("\n=== Validating Scale Structure ===")
    mapper = CrossCulturalMusicMapper()
    
    all_scales = []
    
    # Get all ragas
    for raga_name in mapper.raga_system.ragas.keys():
        raga = mapper.raga_system.get_raga(raga_name)
        if raga:
            all_scales.append(("raga", raga))
    
    # Get all maqamat
    for maqam_name in mapper.maqam_system.maqamat.keys():
        maqam = mapper.maqam_system.get_maqam(maqam_name)
        if maqam:
            all_scales.append(("maqam", maqam))
    
    # Get all pentatonic scales
    for scale_name in mapper.pentatonic_system.scales.keys():
        scale = mapper.pentatonic_system.get_scale(scale_name)
        if scale:
            all_scales.append(("pentatonic", scale))
    
    passed = 0
    failed = 0
    
    for system_name, scale in all_scales:
        errors = []
        
        if not scale.name:
            errors.append("missing name")
        if not scale.intervals_semitones:
            errors.append("missing intervals")
        if not scale.emotion_mapping:
            errors.append("missing emotion_mapping")
        if not scale.emotional_qualities:
            errors.append("missing emotional_qualities")
        if len(scale.intensity_range) != 2:
            errors.append("invalid intensity_range")
        if not scale.therapeutic_use:
            errors.append("missing therapeutic_use")
        
        if errors:
            print(f"  ✗ {system_name}/{scale.name}: {', '.join(errors)}")
            failed += 1
        else:
            passed += 1
    
    print(f"\nStructure validation: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all validations."""
    print("=" * 70)
    print("Cross-Cultural Music Mapping Validation")
    print("=" * 70)
    
    results = []
    results.append(("Raga System", validate_ragas()))
    results.append(("Maqam System", validate_maqamat()))
    results.append(("Pentatonic System", validate_pentatonic()))
    results.append(("Integration", validate_integration()))
    results.append(("Scale Structure", validate_scale_structure()))
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All validations passed!")
        return 0
    else:
        print("\n✗ Some validations failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

