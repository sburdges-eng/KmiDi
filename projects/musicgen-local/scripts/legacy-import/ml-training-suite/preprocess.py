#!/usr/bin/env python3
"""Preprocess audio files and extract features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

from src.utils.audio import load_audio, extract_mel_spectrogram, normalize_audio


def preprocess_audio_files(
    input_dir: Path,
    output_dir: Path,
    sample_rate: int = 22050,
    duration: float = 5.0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    extensions: list = [".mp3", ".wav", ".aif", ".aiff", ".flac"],
):
    """
    Preprocess audio files and save mel spectrograms.

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save preprocessed features
        sample_rate: Target sample rate
        duration: Duration to extract in seconds
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length
        extensions: List of audio file extensions
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_dir.rglob(f"*{ext}"))

    print(f"Found {len(audio_files)} audio files")

    # Process metadata
    metadata = {
        "sample_rate": sample_rate,
        "duration": duration,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "files": [],
    }

    # Process files
    for audio_path in tqdm(audio_files, desc="Processing"):
        try:
            # Load audio
            waveform, sr = load_audio(
                audio_path,
                target_sr=sample_rate,
                mono=True,
                duration=duration,
            )

            # Normalize
            waveform = normalize_audio(waveform, method="peak")

            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(
                waveform,
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            )

            # Create output path (preserve directory structure)
            rel_path = audio_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix(".pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save tensor
            torch.save(mel_spec, output_path)

            # Add to metadata
            metadata["files"].append({
                "input": str(audio_path),
                "output": str(output_path),
                "shape": list(mel_spec.shape),
            })

        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            continue

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProcessed {len(metadata['files'])} files")
    print(f"Saved to: {output_dir}")
    print(f"Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing audio files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed features",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration to extract (seconds)",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel bands",
    )
    args = parser.parse_args()

    preprocess_audio_files(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels,
    )


if __name__ == "__main__":
    main()
