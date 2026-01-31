#!/usr/bin/env python3
"""
Audio Refinery
Processes raw samples with lo-fi bedroom emo aesthetic
Uses audiomentations for transformations
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, \
    LowPassFilter, HighPassFilter, Normalize, TanhDistortion

VAULT_DIR = Path.home() / "Music" / "AudioVault"
RAW_DIR = VAULT_DIR / "raw"
REFINED_DIR = VAULT_DIR / "refined"


# Lo-Fi Bedroom Emo Processing Chain
lofi_chain = Compose([
    # Add tape-like saturation
    TanhDistortion(min_distortion=0.05, max_distortion=0.15, p=0.7),

    # Add warmth with slight low-pass
    LowPassFilter(min_cutoff_freq=8000, max_cutoff_freq=12000, p=0.5),

    # Add slight noise for tape hiss
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.003, p=0.6),

    # Normalize to prevent clipping
    Normalize(p=1.0),
])


def refine_sample(input_path, output_path, sample_rate=44100):
    """Apply lo-fi processing to a sample"""
    # Load audio
    audio, sr = sf.read(input_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Ensure float32
    audio = audio.astype(np.float32)

    # Apply lo-fi chain
    refined = lofi_chain(samples=audio, sample_rate=sr)

    # Save
    sf.write(output_path, refined, sr)


def refine_kick(audio, sr):
    """Specific processing for kicks"""
    chain = Compose([
        TanhDistortion(min_distortion=0.1, max_distortion=0.2, p=0.8),
        LowPassFilter(min_cutoff_freq=150, max_cutoff_freq=250, p=0.5),
        Normalize(p=1.0),
    ])
    return chain(samples=audio, sample_rate=sr)


def refine_snare(audio, sr):
    """Specific processing for snares"""
    chain = Compose([
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.01, p=0.7),
        HighPassFilter(min_cutoff_freq=80, max_cutoff_freq=120, p=0.4),
        TanhDistortion(min_distortion=0.05, max_distortion=0.12, p=0.6),
        Normalize(p=1.0),
    ])
    return chain(samples=audio, sample_rate=sr)


def refine_hihat(audio, sr):
    """Specific processing for hi-hats"""
    chain = Compose([
        HighPassFilter(min_cutoff_freq=3000, max_cutoff_freq=5000, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.002, p=0.5),
        Normalize(p=1.0),
    ])
    return chain(samples=audio, sample_rate=sr)


def detect_sample_type(filename):
    """Detect sample type from filename"""
    filename_lower = filename.lower()

    if "kick" in filename_lower:
        return "kick"
    elif "snare" in filename_lower:
        return "snare"
    elif "hihat" in filename_lower or "hat" in filename_lower:
        return "hihat"
    elif "tom" in filename_lower:
        return "tom"
    elif "crash" in filename_lower or "cymbal" in filename_lower:
        return "cymbal"
    else:
        return "other"


def refine_with_type(input_path, output_path):
    """Refine sample with type-specific processing"""
    # Load audio
    audio, sr = sf.read(input_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Detect type and apply specific chain
    sample_type = detect_sample_type(input_path.name)

    if sample_type == "kick":
        refined = refine_kick(audio, sr)
    elif sample_type == "snare":
        refined = refine_snare(audio, sr)
    elif sample_type == "hihat":
        refined = refine_hihat(audio, sr)
    else:
        # Default lo-fi chain
        refined = lofi_chain(samples=audio, sample_rate=sr)

    # Save
    sf.write(output_path, refined, sr)


def main():
    """Refine all raw samples"""
    print("=" * 60)
    print("AUDIO REFINERY")
    print("Lo-Fi Bedroom Emo Processing")
    print("=" * 60)

    if not RAW_DIR.exists():
        print(f"❌ Raw directory not found: {RAW_DIR}")
        print("   Run generate_demo_samples.py first")
        return

    # Find all audio files in raw/
    audio_files = []
    for ext in ['*.wav', '*.aiff', '*.mp3', '*.flac']:
        audio_files.extend(RAW_DIR.rglob(ext))

    if not audio_files:
        print(f"❌ No audio files found in {RAW_DIR}")
        return

    print(f"\n📁 Found {len(audio_files)} raw samples\n")

    # Process each sample
    refined_count = 0

    for input_path in audio_files:
        # Build output path (preserve folder structure)
        relative_path = input_path.relative_to(RAW_DIR)
        output_path = REFINED_DIR / relative_path

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Refine
        try:
            refine_with_type(input_path, output_path)
            print(f"✅ {relative_path}")
            refined_count += 1
        except Exception as e:
            print(f"❌ {relative_path} - {e}")

    print("\n" + "=" * 60)
    print(f"✅ REFINED {refined_count}/{len(audio_files)} SAMPLES")
    print("=" * 60)
    print(f"\nRefined samples: {REFINED_DIR}")
    print(f"\n📖 Next: python3 build_logic_kit.py")
    print()


if __name__ == "__main__":
    main()
