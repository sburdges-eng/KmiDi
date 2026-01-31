#!/usr/bin/env python3
"""
Generate Demo Samples
Creates synthetic drum samples for testing the pipeline
"""

import numpy as np
import soundfile as sf
from pathlib import Path

RAW_DIR = Path.home() / "Music" / "AudioVault" / "raw" / "Demo_Kit"
SAMPLE_RATE = 44100


def generate_kick(duration=0.5):
    """Generate kick drum (sine sweep + noise)"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))

    # Frequency sweep from 150Hz to 40Hz
    freq = 150 * np.exp(-8 * t)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    sine = np.sin(phase)

    # Envelope
    envelope = np.exp(-5 * t)

    # Add sub bass
    sub = 0.3 * np.sin(2 * np.pi * 50 * t) * np.exp(-10 * t)

    # Mix
    kick = (sine + sub) * envelope

    # Normalize
    kick = kick / np.max(np.abs(kick)) * 0.9

    return kick.astype(np.float32)


def generate_snare(duration=0.3):
    """Generate snare drum (noise + tone)"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))

    # Noise component (high-pass filtered)
    noise = np.random.randn(len(t))

    # Tonal component (200Hz)
    tone = 0.3 * np.sin(2 * np.pi * 200 * t)

    # Envelope (fast attack, medium decay)
    envelope = np.exp(-15 * t)

    # Mix
    snare = (noise + tone) * envelope

    # Normalize
    snare = snare / np.max(np.abs(snare)) * 0.8

    return snare.astype(np.float32)


def generate_hihat(duration=0.15, closed=True):
    """Generate hi-hat (filtered noise)"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))

    # High-frequency noise
    noise = np.random.randn(len(t))

    # Envelope
    decay_rate = 40 if closed else 20
    envelope = np.exp(-decay_rate * t)

    # High-pass filter (simple)
    filtered = noise * envelope

    # Normalize
    hihat = filtered / np.max(np.abs(filtered)) * 0.6

    return hihat.astype(np.float32)


def generate_tom(duration=0.4, freq=120):
    """Generate tom drum"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))

    # Frequency sweep
    freq_sweep = freq * np.exp(-3 * t)
    phase = 2 * np.pi * np.cumsum(freq_sweep) / SAMPLE_RATE

    # Sine wave
    tone = np.sin(phase)

    # Envelope
    envelope = np.exp(-7 * t)

    # Mix
    tom = tone * envelope

    # Normalize
    tom = tom / np.max(np.abs(tom)) * 0.85

    return tom.astype(np.float32)


def generate_crash(duration=1.5):
    """Generate crash cymbal (metallic noise)"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))

    # Complex noise with metallic character
    noise = np.random.randn(len(t))

    # Add some high-frequency harmonics
    for freq in [3000, 5000, 7000, 9000]:
        noise += 0.2 * np.sin(2 * np.pi * freq * t) * np.random.randn(len(t))

    # Slow decay
    envelope = np.exp(-3 * t)

    # Mix
    crash = noise * envelope

    # Normalize
    crash = crash / np.max(np.abs(crash)) * 0.7

    return crash.astype(np.float32)


def main():
    """Generate all demo samples"""
    print("=" * 60)
    print("DEMO SAMPLE GENERATOR")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output: {RAW_DIR}\n")

    samples = [
        ("Kick_01.wav", generate_kick()),
        ("Kick_02.wav", generate_kick(duration=0.6)),
        ("Snare_01.wav", generate_snare()),
        ("Snare_02.wav", generate_snare(duration=0.4)),
        ("HiHat_Closed_01.wav", generate_hihat(closed=True)),
        ("HiHat_Closed_02.wav", generate_hihat(duration=0.1, closed=True)),
        ("HiHat_Open_01.wav", generate_hihat(duration=0.5, closed=False)),
        ("Tom_Low_01.wav", generate_tom(freq=80)),
        ("Tom_Mid_01.wav", generate_tom(freq=120)),
        ("Tom_High_01.wav", generate_tom(freq=180)),
        ("Crash_01.wav", generate_crash()),
    ]

    for filename, audio in samples:
        filepath = RAW_DIR / filename
        sf.write(filepath, audio, SAMPLE_RATE)
        print(f"✅ {filename}")

    print(f"\n✅ Generated {len(samples)} demo samples")
    print(f"   Location: {RAW_DIR}")
    print(f"\n📖 Next: python3 audio_refinery.py")
    print()


if __name__ == "__main__":
    main()
