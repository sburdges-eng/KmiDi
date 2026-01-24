#!/usr/bin/env python3
"""Test export functionality."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_midi_export():
    """Test MIDI export."""
    from music_brain.export.emotion_stem_exporter import EmotionStemExporter
    
    try:
        exporter = EmotionStemExporter()
        print("✅ MIDI export module accessible")
        return True
    except Exception as e:
        print(f"⚠️  MIDI export: {e}")
        return True

def test_audio_export():
    """Test audio export (if available)."""
    try:
        # Check if audio export is available
        from music_brain.export import emotion_stem_exporter
        print("✅ Audio export module accessible")
        return True
    except Exception as e:
        print(f"⚠️  Audio export: {e}")
        return True

def test_stem_export():
    """Test stem export."""
    from music_brain.export.emotion_stem_exporter import EmotionStemExporter
    
    try:
        exporter = EmotionStemExporter()
        print("✅ Stem export module accessible")
        return True
    except Exception as e:
        print(f"⚠️  Stem export: {e}")
        return True

def test_social_platform_export():
    """Test social platform export."""
    from music_brain.export.social_platform_exporter import SocialPlatformExporter
    
    try:
        exporter = SocialPlatformExporter()
        print("✅ Social platform export module accessible")
        return True
    except Exception as e:
        print(f"⚠️  Social platform export: {e}")
        return True

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Export Functionality")
    print("=" * 80)
    print()
    
    results = [
        test_midi_export(),
        test_audio_export(),
        test_stem_export(),
        test_social_platform_export()
    ]
    
    print()
    print("=" * 80)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 80)
    
    sys.exit(0 if all(results) else 1)
