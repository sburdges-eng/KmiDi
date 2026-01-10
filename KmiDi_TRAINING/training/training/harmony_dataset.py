"""
Minimal HarmonyPredictor dataset loader.

Manifest format (jsonl):
{"context": [0.1, 0.2, ...], "target": [0.3, 0.4, ...]}

Both vectors should be numeric lists; lengths must match model expectations
(default: context 128, target 64).
"""

import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset


class HarmonyDataset(Dataset):
    def __init__(self, manifest_path: str, expected_context: int = 128, expected_target: int = 64):
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self.samples: List[Dict] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "context" in obj and "target" in obj:
                        self.samples.append(obj)
                except json.JSONDecodeError:
                    continue
        if not self.samples:
            raise ValueError(f"No valid samples in manifest: {manifest_path}")

        self.expected_context = expected_context
        self.expected_target = expected_target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ctx = torch.tensor(sample["context"], dtype=torch.float32)
        tgt = torch.tensor(sample["target"], dtype=torch.float32)

        # Pad/truncate to expected sizes
        if ctx.numel() < self.expected_context:
            ctx = torch.nn.functional.pad(ctx, (0, self.expected_context - ctx.numel()))
        ctx = ctx[: self.expected_context]

        if tgt.numel() < self.expected_target:
            tgt = torch.nn.functional.pad(tgt, (0, self.expected_target - tgt.numel()))
        tgt = tgt[: self.expected_target]

        return ctx, tgt


__all__ = ["HarmonyDataset"]
