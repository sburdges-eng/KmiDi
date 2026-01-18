#!/usr/bin/env python3
"""
Dataset Loaders for KmiDi Music Brain
======================================

Provides PyTorch DataLoaders for:
- Lakh MIDI dataset (melody, harmony, groove)
- M4Singer dataset (emotion, dynamics)

Usage:
    from dataset_loaders import get_lakh_midi_loader, get_m4singer_loader

    train_loader = get_lakh_midi_loader(
        data_dir=os.path.join(default_data_root(), "raw/chord_progressions/lakh_midi"),
        task="melody",
        batch_size=32
    )
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
# Default dataset root
def default_data_root() -> str:
    return (
        os.environ.get("KMI_DI_AUDIO_DATA_ROOT")
        or os.environ.get("KELLY_AUDIO_DATA_ROOT")
        or str(Path(__file__).resolve().parent.parent / "data" / "audio")
    )

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Optional imports - gracefully handle missing dependencies
try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@dataclass
class MIDIFeatures:
    """Features extracted from a MIDI file."""
    pitches: np.ndarray
    velocities: np.ndarray
    durations: np.ndarray
    start_times: np.ndarray
    chords: Optional[np.ndarray] = None
    tempo: float = 120.0


class LakhMIDIDataset(Dataset):
    """
    PyTorch Dataset for Lakh MIDI files.

    Supports different tasks:
    - melody: Note sequences for melody generation
    - harmony: Chord progressions for harmony prediction
    - groove: Timing patterns for groove prediction
    """

    # MIDI pitch range
    MIN_PITCH = 21  # A0
    MAX_PITCH = 108  # C8
    NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1

    # Chord vocabulary (simplified)
    CHORD_TYPES = ['C', 'Cm', 'C7', 'Cmaj7', 'Cm7', 'Cdim', 'Caug',
                   'D', 'Dm', 'D7', 'Dmaj7', 'Dm7', 'Ddim', 'Daug',
                   'E', 'Em', 'E7', 'Emaj7', 'Em7', 'Edim', 'Eaug',
                   'F', 'Fm', 'F7', 'Fmaj7', 'Fm7', 'Fdim', 'Faug',
                   'G', 'Gm', 'G7', 'Gmaj7', 'Gm7', 'Gdim', 'Gaug',
                   'A', 'Am', 'A7', 'Amaj7', 'Am7', 'Adim', 'Aaug',
                   'B', 'Bm', 'B7', 'Bmaj7', 'Bm7', 'Bdim', 'Baug']

    def __init__(
        self,
        data_dir: str,
        task: str = "melody",
        seq_length: int = 128,
        split: str = "train",
        max_files: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the Lakh MIDI dataset.

        Args:
            data_dir: Path to lakh_midi directory
            task: One of "melody", "harmony", "groove"
            seq_length: Sequence length for padding/truncation
            split: One of "train", "val", "test"
            max_files: Maximum number of files to load (for testing)
            cache_dir: Directory to cache processed features
        """
        if not HAS_PRETTY_MIDI:
            raise ImportError("pretty_midi required: pip install pretty_midi")

        self.data_dir = Path(data_dir)
        self.task = task
        self.seq_length = seq_length
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"

        # Find all MIDI files
        self.midi_files = self._find_midi_files(max_files)

        # Split data
        self._apply_split()

        print(f"LakhMIDIDataset: {len(self.midi_files)} files for {split} ({task})")

    def _find_midi_files(self, max_files: Optional[int]) -> List[Path]:
        """Find all MIDI files in the data directory."""
        midi_files = []

        # Check for lmd_full subdirectory
        lmd_full = self.data_dir / "lmd_full"
        search_dir = lmd_full if lmd_full.exists() else self.data_dir

        for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
            midi_files.extend(search_dir.rglob(ext))

        # Shuffle and limit
        random.seed(42)
        random.shuffle(midi_files)

        if max_files:
            midi_files = midi_files[:max_files]

        return midi_files

    def _apply_split(self):
        """Apply train/val/test split."""
        n = len(self.midi_files)

        if self.split == "train":
            self.midi_files = self.midi_files[:int(0.8 * n)]
        elif self.split == "val":
            self.midi_files = self.midi_files[int(0.8 * n):int(0.9 * n)]
        else:  # test
            self.midi_files = self.midi_files[int(0.9 * n):]

    def _extract_features(self, midi_path: Path) -> Optional[MIDIFeatures]:
        """Extract features from a MIDI file."""
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))

            # Get all notes from all instruments
            notes = []
            # For groove prediction, include drum tracks (they define timing/groove)
            # For melody/harmony tasks, skip drums
            include_drums = self.task == "groove_prediction"

            for instrument in midi.instruments:
                if instrument.is_drum and not include_drums:
                    continue  # Skip drums for melody/harmony tasks
                for note in instrument.notes:
                    notes.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'duration': note.end - note.start
                    })

            # Fallback: if no pitched notes, try drums anyway for any task
            if not notes:
                for instrument in midi.instruments:
                    if instrument.is_drum:
                        for note in instrument.notes:
                            notes.append({
                                'pitch': note.pitch,
                                'velocity': note.velocity,
                                'start': note.start,
                                'duration': note.end - note.start
                            })

            if not notes:
                return None

            # Sort by start time
            notes.sort(key=lambda x: x['start'])

            # Extract arrays
            pitches = np.array([n['pitch'] for n in notes], dtype=np.int64)
            velocities = np.array([n['velocity'] for n in notes], dtype=np.float32) / 127.0
            durations = np.array([n['duration'] for n in notes], dtype=np.float32)
            start_times = np.array([n['start'] for n in notes], dtype=np.float32)

            # Normalize pitches to vocab range
            pitches = np.clip(pitches - self.MIN_PITCH, 0, self.NUM_PITCHES - 1)

            # Get tempo
            tempo = midi.estimate_tempo() if midi.estimate_tempo() else 120.0

            return MIDIFeatures(
                pitches=pitches,
                velocities=velocities,
                durations=durations,
                start_times=start_times,
                tempo=tempo
            )

        except Exception as e:
            return None

    def _get_melody_features(self, features: MIDIFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get features for melody generation task."""
        # Input: sequence of pitches as float, Output: next pitch class
        pitches = features.pitches[:self.seq_length + 1]

        # Pad if needed
        if len(pitches) < self.seq_length + 1:
            pitches = np.pad(pitches, (0, self.seq_length + 1 - len(pitches)))

        # Convert to float and normalize for LSTM input
        # Shape: (seq_length, 1) -> will be (batch, seq_length, 1) after batching
        x_pitches = pitches[:-1].astype(np.float32) / self.NUM_PITCHES
        x = torch.tensor(x_pitches, dtype=torch.float32).unsqueeze(-1)  # Add feature dim

        # Target: next pitch as class index
        y = torch.tensor(pitches[-1], dtype=torch.long)

        return x, y

    def _get_harmony_features(self, features: MIDIFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get features for harmony prediction task."""
        # Simplified: predict chord from pitch context
        pitches = features.pitches[:self.seq_length]

        # Pad if needed
        if len(pitches) < self.seq_length:
            pitches = np.pad(pitches, (0, self.seq_length - len(pitches)))

        # One-hot encode pitches
        x = np.zeros((self.seq_length, self.NUM_PITCHES), dtype=np.float32)
        for i, p in enumerate(pitches):
            if p < self.NUM_PITCHES:
                x[i, p] = 1.0

        # Flatten for MLP
        x = x.flatten()

        # Target: simplified chord based on most common pitch class
        pitch_classes = pitches % 12
        most_common = np.bincount(pitch_classes, minlength=12).argmax()
        y = most_common  # 0-11 for pitch class, extend to 48 for chord types

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _get_groove_features(self, features: MIDIFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get features for groove prediction task."""
        # Input: rhythmic pattern, Output: micro-timing offsets
        durations = features.durations[:self.seq_length]
        start_times = features.start_times[:self.seq_length]

        # Pad if needed
        if len(durations) < self.seq_length:
            durations = np.pad(durations, (0, self.seq_length - len(durations)))
            start_times = np.pad(start_times, (0, self.seq_length - len(start_times)))

        # Normalize
        durations = durations / (durations.max() + 1e-6)

        # Calculate inter-onset intervals
        ioi = np.diff(start_times, prepend=0)
        ioi = ioi / (ioi.max() + 1e-6)

        # Stack features
        x = np.stack([durations, ioi], axis=-1).astype(np.float32)

        # Target: predict next timing offset
        y = ioi[-32:] if len(ioi) >= 32 else np.pad(ioi, (0, 32 - len(ioi)))

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.midi_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        midi_path = self.midi_files[idx]

        # Try to extract features
        features = self._extract_features(midi_path)

        if features is None:
            # Return dummy data on failure
            if self.task == "melody":
                return torch.zeros(self.seq_length, 1, dtype=torch.float32), torch.tensor(0, dtype=torch.long)
            elif self.task == "harmony":
                return torch.zeros(self.seq_length * self.NUM_PITCHES, dtype=torch.float32), torch.tensor(0, dtype=torch.long)
            else:  # groove
                return torch.zeros(self.seq_length, 2, dtype=torch.float32), torch.zeros(32, dtype=torch.float32)

        # Get task-specific features
        if self.task == "melody":
            return self._get_melody_features(features)
        elif self.task == "harmony":
            return self._get_harmony_features(features)
        else:  # groove
            return self._get_groove_features(features)


class M4SingerDataset(Dataset):
    """
    PyTorch Dataset for M4Singer audio files.

    Supports different tasks:
    - emotion: Emotion classification from audio
    - dynamics: Expression parameter extraction
    """

    EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']

    def __init__(
        self,
        data_dir: str,
        task: str = "emotion",
        sample_rate: int = 16000,
        duration: float = 5.0,
        split: str = "train",
        max_files: Optional[int] = None
    ):
        """
        Initialize the M4Singer dataset.

        Args:
            data_dir: Path to m4singer directory
            task: One of "emotion", "dynamics"
            sample_rate: Audio sample rate
            duration: Clip duration in seconds
            split: One of "train", "val", "test"
            max_files: Maximum number of files to load
        """
        if not HAS_LIBROSA:
            raise ImportError("librosa required: pip install librosa")

        self.data_dir = Path(data_dir)
        self.task = task
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.split = split

        # Find audio files
        self.audio_files = self._find_audio_files(max_files)
        self._apply_split()

        print(f"M4SingerDataset: {len(self.audio_files)} files for {split} ({task})")

    def _find_audio_files(self, max_files: Optional[int]) -> List[Path]:
        """Find all audio files."""
        audio_files = []

        for ext in ["*.wav", "*.mp3", "*.flac"]:
            audio_files.extend(self.data_dir.rglob(ext))

        random.seed(42)
        random.shuffle(audio_files)

        if max_files:
            audio_files = audio_files[:max_files]

        return audio_files

    def _apply_split(self):
        """Apply train/val/test split."""
        n = len(self.audio_files)

        if self.split == "train":
            self.audio_files = self.audio_files[:int(0.8 * n)]
        elif self.split == "val":
            self.audio_files = self.audio_files[int(0.8 * n):int(0.9 * n)]
        else:
            self.audio_files = self.audio_files[int(0.9 * n):]

    def _load_audio(self, audio_path: Path) -> Optional[np.ndarray]:
        """Load and preprocess audio file."""
        try:
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=self.duration)

            # Pad or truncate
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)))
            else:
                audio = audio[:self.n_samples]

            return audio

        except Exception:
            return None

    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=256
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        return mel_db.astype(np.float32)

    def _get_emotion_features(self, audio: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get features for emotion classification."""
        mel = self._extract_mel_spectrogram(audio)

        # Add channel dimension for CNN
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        # Generate pseudo-label based on audio features (for demo)
        # In real usage, load actual labels from metadata
        energy = np.mean(np.abs(audio))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        # Simple heuristic for demo
        if energy > 0.1 and zcr > 0.1:
            label = 0  # happy
        elif energy < 0.05:
            label = 1  # sad
        else:
            label = 6  # neutral

        return x, torch.tensor(label, dtype=torch.long)

    def _get_dynamics_features(self, audio: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get features for dynamics/expression extraction."""
        # Extract various audio features
        rms = librosa.feature.rms(y=audio)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]

        # Normalize and combine
        rms = (rms - rms.mean()) / (rms.std() + 1e-6)
        spectral_centroid = (spectral_centroid - spectral_centroid.mean()) / (spectral_centroid.std() + 1e-6)

        # Pad/truncate to fixed size
        target_len = 32
        rms = np.pad(rms, (0, max(0, target_len - len(rms))))[:target_len]
        spectral_centroid = np.pad(spectral_centroid, (0, max(0, target_len - len(spectral_centroid))))[:target_len]

        x = torch.tensor(rms, dtype=torch.float32)
        y = torch.tensor(spectral_centroid[:16], dtype=torch.float32)

        return x, y

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.audio_files[idx]
        audio = self._load_audio(audio_path)

        if audio is None:
            # Return dummy data
            if self.task == "emotion":
                return torch.zeros(1, 64, 128, dtype=torch.float32), torch.tensor(0, dtype=torch.long)
            else:
                return torch.zeros(32, dtype=torch.float32), torch.zeros(16, dtype=torch.float32)

        if self.task == "emotion":
            return self._get_emotion_features(audio)
        else:
            return self._get_dynamics_features(audio)


def get_lakh_midi_loader(
    data_dir: str,
    task: str = "melody",
    batch_size: int = 32,
    split: str = "train",
    max_files: Optional[int] = None,
    num_workers: int = 0
) -> DataLoader:
    """
    Get a DataLoader for the Lakh MIDI dataset.

    Args:
        data_dir: Path to lakh_midi directory
        task: One of "melody", "harmony", "groove"
        batch_size: Batch size
        split: One of "train", "val", "test"
        max_files: Maximum number of files
        num_workers: Number of data loading workers

    Returns:
        PyTorch DataLoader
    """
    dataset = LakhMIDIDataset(
        data_dir=data_dir,
        task=task,
        split=split,
        max_files=max_files
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=False
    )


def get_m4singer_loader(
    data_dir: str,
    task: str = "emotion",
    batch_size: int = 32,
    split: str = "train",
    max_files: Optional[int] = None,
    num_workers: int = 0
) -> DataLoader:
    """
    Get a DataLoader for the M4Singer dataset.

    Args:
        data_dir: Path to m4singer directory
        task: One of "emotion", "dynamics"
        batch_size: Batch size
        split: One of "train", "val", "test"
        max_files: Maximum number of files
        num_workers: Number of data loading workers

    Returns:
        PyTorch DataLoader
    """
    dataset = M4SingerDataset(
        data_dir=data_dir,
        task=task,
        split=split,
        max_files=max_files
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=False
    )


if __name__ == "__main__":
    # Test the loaders
    import argparse

    parser = argparse.ArgumentParser()
    # Updated: Files moved from external SSD to local storage (2025-01-09)
    parser.add_argument("--data-dir", default=default_data_root())
    parser.add_argument("--dataset", choices=["lakh", "m4singer"], default="lakh")
    parser.add_argument("--task", default="melody")
    parser.add_argument("--max-files", type=int, default=100)
    args = parser.parse_args()

    if args.dataset == "lakh":
        loader = get_lakh_midi_loader(
            data_dir=f"{args.data_dir}/lakh_midi",
            task=args.task,
            max_files=args.max_files
        )
    else:
        loader = get_m4singer_loader(
            data_dir=f"{args.data_dir}/m4singer",
            task=args.task,
            max_files=args.max_files
        )

    print(f"\nTesting {args.dataset} loader for {args.task}...")
    for batch_idx, (x, y) in enumerate(loader):
        print(f"Batch {batch_idx}: x={x.shape}, y={y.shape}")
        if batch_idx >= 2:
            break
    print("Done!")
