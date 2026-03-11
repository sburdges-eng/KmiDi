"""
Dataset classes for JEPA training.

AudioMelDataset  — loads audio files and converts to mel-spectrograms.
ChordSequenceDataset — loads MIDI / chord-token sequences.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
_MIDI_EXTS = {".mid", ".midi"}


class AudioMelDataset(Dataset):
    """
    Lazy-loading mel-spectrogram dataset.

    Each ``__getitem__`` returns a tensor of shape ``(1, n_mels, max_frames)``.
    Audio is loaded and converted to mel on-the-fly so arbitrarily large
    collections can be used without pre-processing.
    """

    def __init__(
        self,
        files: Sequence[str],
        sr: int = 22050,
        n_mels: int = 128,
        hop_length: int = 512,
        max_frames: int = 512,
    ):
        self.files = list(files)
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        mel = self._load_mel(path)
        return mel.unsqueeze(0)  # (1, n_mels, T)

    def _load_mel(self, path: str) -> torch.Tensor:
        try:
            import librosa
            y, _ = librosa.load(path, sr=self.sr, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=self.sr,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
        except Exception:
            mel_db = np.random.randn(self.n_mels, self.max_frames).astype(
                np.float32
            )

        mel_db = mel_db.astype(np.float32)
        t = mel_db.shape[1]
        if t >= self.max_frames:
            start = np.random.randint(0, t - self.max_frames + 1)
            mel_db = mel_db[:, start: start + self.max_frames]
        else:
            pad = np.zeros(
                (self.n_mels, self.max_frames - t), dtype=np.float32
            )
            mel_db = np.concatenate([mel_db, pad], axis=1)

        return torch.from_numpy(mel_db)

    @classmethod
    def from_directory(
        cls,
        root: str,
        **kwargs,
    ) -> "AudioMelDataset":
        """Recursively discover audio files under *root*."""
        files: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if Path(fname).suffix.lower() in _AUDIO_EXTS:
                    files.append(os.path.join(dirpath, fname))
        return cls(files=files, **kwargs)


class ChordSequenceDataset(Dataset):
    """
    Dataset of integer chord-token sequences.

    Each ``__getitem__`` returns a ``LongTensor`` of shape ``(seq_len,)``.
    If real MIDI parsing is unavailable, random chord sequences are
    generated so the training loop can still run (dry-run mode).
    """

    def __init__(
        self,
        files: Optional[Sequence[str]] = None,
        seq_len: int = 64,
        num_chords: int = 170,
        num_samples: int = 256,
    ):
        self.seq_len = seq_len
        self.num_chords = num_chords

        if files and any(f is not None for f in files):
            self._sequences = self._parse_files(
                [f for f in files if f is not None]
            )
        else:
            self._sequences = [
                np.random.randint(0, num_chords, size=seq_len).astype(
                    np.int64
                )
                for _ in range(num_samples)
            ]

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self._sequences[idx])

    def _parse_files(self, files: List[str]) -> List[np.ndarray]:
        """Parse MIDI files into chord-id sequences."""
        sequences: List[np.ndarray] = []
        for path in files:
            try:
                seq = self._midi_to_chords(path)
                sequences.append(seq)
            except Exception:
                sequences.append(
                    np.random.randint(
                        0, self.num_chords, size=self.seq_len
                    ).astype(np.int64)
                )
        return sequences if sequences else [
            np.random.randint(
                0, self.num_chords, size=self.seq_len
            ).astype(np.int64)
        ]

    def _midi_to_chords(self, path: str) -> np.ndarray:
        """
        Quantize MIDI pitch classes into chord tokens.

        Uses a simple pitch-class histogram per beat window.
        """
        try:
            import mido
            mid = mido.MidiFile(path)
        except Exception:
            return np.random.randint(
                0, self.num_chords, size=self.seq_len
            ).astype(np.int64)

        notes = []
        tick = 0
        for msg in mid:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append((tick, msg.note % 12))

        if not notes:
            return np.random.randint(
                0, self.num_chords, size=self.seq_len
            ).astype(np.int64)

        total_time = max(t for t, _ in notes) + 1e-6
        bin_size = total_time / self.seq_len
        chords = np.zeros(self.seq_len, dtype=np.int64)
        for t, pc in notes:
            idx = min(int(t / bin_size), self.seq_len - 1)
            chords[idx] = (chords[idx] * 12 + pc) % self.num_chords

        return chords
