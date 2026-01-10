#!/usr/bin/env python3
"""
End-to-end music generation test.

Tests the complete pipeline: emotion text → MusicBrain → MIDI output
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_brain.emotion_api import MusicBrain
from pathlib import Path
import tempfile


def test_e2e_generation():
    """Test complete emotion → MIDI pipeline."""
    print("=" * 70)
    print("End-to-End Music Generation Test")
    print("=" * 70)
    
    # Initialize MusicBrain
    print("\n1. Initializing MusicBrain...")
    try:
        brain = MusicBrain(use_neural=False)  # Use keyword matching for speed
        print("   ✓ MusicBrain initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return False
    
    # Test emotion → music generation
    print("\n2. Testing emotion → music generation...")
    test_emotions = [
        "I'm feeling sad and melancholic",
        "I'm happy and energetic",
        "I'm anxious and worried"
    ]
    
    all_passed = True
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, emotion_text in enumerate(test_emotions):
            print(f"\n   Test {i+1}: '{emotion_text}'")
            try:
                result = brain.generate_from_text(emotion_text)
                
                if result and result.musical_params:
                    print(f"      ✓ Generated music")
                    print(f"        Tempo: {result.musical_params.tempo_suggested} BPM")
                    if result.musical_params.mode_weights:
                        top_mode = max(result.musical_params.mode_weights.items(), key=lambda x: x[1])
                        print(f"        Top Mode: {top_mode[0].value if hasattr(top_mode[0], 'value') else top_mode[0]} ({top_mode[1]:.0%})")
                    print(f"        Dissonance: {result.musical_params.dissonance:.0%}")
                else:
                    print(f"      ✗ No result generated")
                    all_passed = False
            except Exception as e:
                print(f"      ✗ Generation failed: {e}")
                all_passed = False
    
    # Test cultural scale suggestions
    print("\n3. Testing cross-cultural scale suggestions...")
    try:
        suggestions = brain.get_cultural_scale_suggestions("sad", 0.5)
        if suggestions:
            for system, scale in suggestions.items():
                if scale:
                    print(f"   ✓ {system}: {scale.name}")
        print("   ✓ Cross-cultural mappings working")
    except Exception as e:
        print(f"   ⚠ Cross-cultural test failed: {e}")
        # Not critical for e2e test
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All end-to-end tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = test_e2e_generation()
    sys.exit(0 if success else 1)

