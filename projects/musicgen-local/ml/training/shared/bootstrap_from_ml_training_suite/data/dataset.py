"""Dataset classes for audio ML training."""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import json

from ..utils.audio import load_audio, extract_mel_spectrogram, pad_or_trim, normalize_audio


class AudioDataset(Dataset):
    """
    PyTorch Dataset for audio files.

    Supports loading audio files and extracting features (mel spectrograms, MFCCs, etc.)
    """

    def __init__(
        self,
        audio_paths: List[str | Path],
        labels: Optional[List[int]] = None,
        sample_rate: int = 22050,
        duration: float = 5.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Args:
            audio_paths: List of paths to audio files
            labels: Optional list of integer labels
            sample_rate: Target sample rate
            duration: Duration in seconds to load
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for spectrogram
            transform: Optional transform for features
            target_transform: Optional transform for labels
        """
        self.audio_paths = [Path(p) for p in audio_paths]
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform
        self.target_transform = target_transform

        # Calculate target samples
        self.target_samples = int(sample_rate * duration)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_paths[idx]

        # Load audio
        waveform, sr = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            mono=True,
            duration=self.duration,
        )

        # Normalize
        waveform = normalize_audio(waveform, method="peak")

        # Pad or trim to exact length
        waveform = pad_or_trim(waveform, self.target_samples)

        # Extract mel spectrogram
        features = extract_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Apply transforms
        if self.transform:
            features = self.transform(features)

        # Get label
        label = self.labels[idx] if self.labels is not None else 0
        if self.target_transform:
            label = self.target_transform(label)

        return features, label

    @classmethod
    def from_directory(
        cls,
        root_dir: str | Path,
        extensions: List[str] = [".mp3", ".wav", ".aif", ".aiff", ".flac"],
        **kwargs,
    ) -> "AudioDataset":
        """
        Create dataset from a directory of audio files.

        If subdirectories exist, they are treated as class labels.

        Args:
            root_dir: Root directory containing audio files
            extensions: List of audio file extensions to include
            **kwargs: Additional arguments for AudioDataset

        Returns:
            AudioDataset instance
        """
        root_dir = Path(root_dir)
        audio_paths = []
        labels = []
        class_to_idx = {}

        # Check if subdirectories exist (class-based structure)
        subdirs = [d for d in root_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Class-based directory structure
            for class_idx, class_dir in enumerate(sorted(subdirs)):
                class_name = class_dir.name
                class_to_idx[class_name] = class_idx

                for ext in extensions:
                    for audio_file in class_dir.glob(f"*{ext}"):
                        audio_paths.append(audio_file)
                        labels.append(class_idx)
        else:
            # Flat directory structure (no labels)
            for ext in extensions:
                audio_paths.extend(root_dir.glob(f"*{ext}"))
            labels = None

        dataset = cls(audio_paths=audio_paths, labels=labels, **kwargs)
        dataset.class_to_idx = class_to_idx if subdirs else {}
        dataset.idx_to_class = {v: k for k, v in class_to_idx.items()} if subdirs else {}

        return dataset


def create_dataloaders(
    dataset: AudioDataset,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from a dataset.

    Args:
        dataset: AudioDataset instance
        batch_size: Batch size
        train_split: Fraction for training
        val_split: Fraction for validation (rest is test)
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Calculate split sizes
    total = len(dataset)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    test_size = total - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
