"""
Unified audio emotion dataset for DEAM + EMO-Music manifests.

Manifest schema (JSON list):
{
  "id": "track_001",
  "dataset": "DEAM" | "EMO_Music",
  "audio_path": "datasets/DEAM/audio/001.wav",
  "valence": 0.73,
  "arousal": 0.41,
  "split": "train" | "val" | "test"
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset


class UnifiedEmotionDataset(Dataset):
    def __init__(self, manifest_path: Path, split: Optional[str] = None, target_sr: int = 22050):
        """
        Args:
            manifest_path: Path to manifest JSON.
            split: Optional split filter ("train", "val", "test"). If None, use all.
            target_sr: Target sample rate for resampling.
        """
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            items: List[Dict[str, Any]] = json.load(f)

        if split:
            items = [i for i in items if i.get("split") == split]

        self.items = items
        self.target_sr = target_sr
        # Preload and resample all audio once to avoid doing this in __getitem__
        self._audio: List[torch.Tensor] = []
        self._preload_audio()

    def _preload_audio(self) -> None:
        """Load and resample all audio to target_sr during dataset initialization."""
        for item in self.items:
            audio_path = Path(item["audio_path"])
            wav, sr = torchaudio.load(audio_path)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            self._audio.append(wav)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary with audio tensor (resampled to target_sr) and labels.
        Audio is preloaded in __init__ to avoid per-iteration disk I/O.
        """
        item = self.items[idx]
        wav = self._audio[idx]

        return {
            "audio": wav,
            "sr": self.target_sr,
            "valence": torch.tensor(item["valence"], dtype=torch.float32),
            "arousal": torch.tensor(item["arousal"], dtype=torch.float32),
            "id": item["id"],
            "dataset": item.get("dataset"),
            "split": item.get("split"),
        }
