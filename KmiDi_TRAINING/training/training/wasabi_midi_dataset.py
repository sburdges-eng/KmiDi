"""
WASABI-backed dataset for MIDI generator training.

Uses WASABI-format manifests (emotion, chords, lyrics) and generates
synthetic MIDI token sequences from chords + emotion for the
EmotionMIDITransformer. Output format matches MIDIDataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

# Emotion vocab for mapping lyrics_emotion dict -> 64-dim vector
EMOTION_VOCAB = [
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
    "joy",
    "sorrow",
    "love",
    "hate",
    "excitement",
    "calm",
    "anxiety",
    "peace",
    "energy",
    "melancholy",
]


def _lyrics_emotion_to_64(lyrics_emotion: Dict[str, float]) -> np.ndarray:
    """Map lyrics_emotion dict to 64-dim vector."""
    out = np.zeros(64, dtype=np.float32)
    for i, e in enumerate(EMOTION_VOCAB):
        if i >= 64:
            break
        v = lyrics_emotion.get(e, 0.0)
        for k, val in lyrics_emotion.items():
            if e in k.lower() or k.lower() in e:
                v = max(v, float(val))
        out[i] = v
    n = np.linalg.norm(out)
    if n > 0:
        out = out / n
    return out


def _chord_root_to_midi(chord: str) -> int:
    """Map chord root to MIDI note (C4=60). Simple mapping."""
    s = chord.strip().upper()
    roots = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}
    for r, n in roots.items():
        if s.startswith(r):
            if "#" in s[:3]:
                return min(96, n + 1)
            if "B" in s[:2] and r == "B":
                return min(96, n - 1)
            return n
    return 60


class WasabiMIDIDataset(Dataset):
    """
    Dataset for MIDI generator training using WASABI-style manifests.

    Reads JSONL with id, lyrics, lyrics_emotion, chords, tempo, etc.
    Generates synthetic MIDI token sequences from chords + emotion.
    Returns input_ids, labels, emotion (64-dim) like MIDIDataset.
    """

    def __init__(
        self,
        manifest_path: str,
        config: dict,
        tokenizer: Any,  # MIDITokenizer
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_len = config.get("max_seq_length", 512)

        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"WASABI manifest not found: {manifest_path}")

        self.samples: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        print(f"[WasabiMIDIDataset] Loaded {len(self.samples)} samples from {manifest_path}")

    def _synthetic_midi_from_sample(self, sample: Dict[str, Any]) -> List[int]:
        """Generate MIDI token sequence from WASABI sample (chords + emotion)."""
        emotions = sample.get("lyrics_emotion", {})
        chords = sample.get("chords", ["C", "F", "G", "C"])
        tempo = max(60, min(180, int(sample.get("tempo", 120))))

        # Emotion-derived params
        happy = emotions.get("happy", 0) + emotions.get("joy", 0)
        sad = emotions.get("sad", 0) + emotions.get("sorrow", 0)
        arousal = (
            emotions.get("excitement", 0) + emotions.get("energy", 0) - emotions.get("calm", 0)
        )
        arousal = max(0, min(1, 0.5 + arousal * 0.5))
        valence = (happy - sad) * 0.5 + 0.5
        valence = max(0, min(1, valence))
        intensity = min(1.0, sum(emotions.values()) / 3.0) if emotions else 0.5

        events = []
        n_bars = max(2, min(8, 2 + int(arousal * 6)))
        n_notes_per_bar = max(2, min(8, int(2 + 4 * intensity + 2 * arousal)))

        for bar in range(n_bars):
            events.append({"type": "bar"})
            base = _chord_root_to_midi(chords[bar % len(chords)])
            # Valence: higher -> slightly higher register
            base = base + int(6 * (valence - 0.5))
            base = max(36, min(84, base))

            for i in range(n_notes_per_bar):
                note = base + (i % 5) - 2
                note = max(24, min(96, note))
                vel = int(40 + 60 * intensity + np.random.randint(-10, 10))
                vel = max(1, min(127, vel))
                t = i / max(1, n_notes_per_bar) + np.random.uniform(-0.03, 0.03)
                t = max(0, min(1, t))
                events.append(
                    {
                        "type": "note_on",
                        "note": note,
                        "velocity": vel,
                        "time_in_bar": t,
                    }
                )

        return self.tokenizer.encode(events)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        emotion_64 = _lyrics_emotion_to_64(sample.get("lyrics_emotion", {}))
        tokens = self._synthetic_midi_from_sample(sample)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
        else:
            tokens = tokens + [self.tokenizer.pad_token] * (self.max_seq_len - len(tokens))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        emotion = torch.tensor(emotion_64, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "emotion": emotion,
        }
