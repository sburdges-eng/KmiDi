#!/usr/bin/env python3
"""
Quick Start: iDAW Tier 1 (Pretrained MIDI/Audio/Voice)

Generates complete music from a single emotion description in <5 minutes.

Usage:
    python quickstart_tier1.py

Optional:
    python quickstart_tier1.py --emotion JOY --duration 16
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_brain.tier1 import Tier1MIDIGenerator, Tier1AudioGenerator, Tier1VoiceGenerator


def main():
    print("\n" + "="*70)
    print("iDAW Tier 1 Quick Start: Pretrained Music Generation")
    print("="*70)

    # Step 1: Create dummy emotion embedding
    print("\n[1/4] Creating emotion embedding...")
    emotion_embedding = np.array([
        # Valence (positive/negative) - first 32 dims
        *np.linspace(-0.8, 0, 32),  # Negative valence (grief)
        # Arousal (energy) - next 32 dims
        *np.linspace(-0.6, 0, 32),  # Low arousal (calm)
    ]).astype(np.float32)
    print(f"✓ Emotion shape: {emotion_embedding.shape}")

    # Step 2: Generate MIDI
    print("\n[2/4] Generating MIDI (melody, harmony, groove)...")
    midi_gen = Tier1MIDIGenerator(device="mps", verbose=True)
    midi_result = midi_gen.full_pipeline(emotion_embedding, length=32)
    print(f"✓ Generated {len(midi_result['melody'])} notes")
    print(f"  Melody (first 8): {midi_result['melody'][:8]}")
    print(f"  Groove swing: {midi_result['groove'].get('swing', 0):.3f}")

    # Step 3: Generate audio
    print("\n[3/4] Synthesizing audio...")
    audio_gen = Tier1AudioGenerator(device="mps", verbose=True)
    audio = audio_gen.synthesize_texture(
        midi_result['melody'],
        midi_result['groove'],
        emotion_embedding,
        duration_seconds=4.0
    )
    print(f"✓ Generated {len(audio)} samples ({len(audio)/22050:.1f} sec @ 22050 Hz)")

    # Step 4: Generate voice
    print("\n[4/4] Generating voice guidance...")
    voice_gen = Tier1VoiceGenerator(device="mps", verbose=True)
    voice_text = "Your feelings are valid. This music honors your grief."
    voice = voice_gen.speak_emotion(voice_text, emotion="calm")
    print(f"✓ Generated voice: {len(voice)} samples")

    # Summary
    print("\n" + "="*70)
    print("✓ Complete! Generated:")
    print(f"  - MIDI:  {len(midi_result['melody'])} notes")
    print(f"  - Audio: {len(audio)} samples (22050 Hz)")
    print(f"  - Voice: {len(voice)} samples")
    print(f"  Device: {midi_gen.device}")
    print("="*70 + "\n")

    return midi_result, audio, voice


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
