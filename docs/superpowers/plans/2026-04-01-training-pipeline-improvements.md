# Training Pipeline Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an emotion probe on frozen JEPA latents to replace the hardcoded linear mapping, then improve JEPA training with emotion auxiliary loss and longer training.

**Architecture:** Phase A: emotion probe MLP (256→128→64→2) trained on cached JEPA embeddings from DEAM/PMEmo music emotion datasets, exported to ONNX, loaded as second model in AudioEmotionRunner. Phase B: add emotion auxiliary loss to JEPA trainer, retrain encoder, re-export.

**Tech Stack:** Python, PyTorch, ONNX, librosa, torchaudio, existing UnifiedEmotionDataset

---

## File Structure

| File | Responsibility |
|------|---------------|
| `music_brain/jepa/emotion_probe.py` | Probe model definition (EmotionProbe) |
| `scripts/build_emotion_manifest.py` | Build manifest JSON from DEAM/PMEmo audio files |
| `scripts/extract_jepa_embeddings.py` | Extract and cache JEPA latents for all emotion-labeled audio |
| `scripts/train_emotion_probe.py` | Train probe on cached embeddings |
| `scripts/export_emotion_probe.py` | Export trained probe to ONNX |
| `tests/unit/test_emotion_probe.py` | Tests for probe model and training |
| `music_brain/jepa/trainer.py` | Modify: add emotion auxiliary loss |
| `config/jepa_training.yaml` | Modify: add emotion_loss_weight, probe_checkpoint |
| `src/ml/AudioEmotionRunner.cpp` | Modify: load + run probe ONNX |
| `include/penta/ml/AudioEmotionRunner.h` | Modify: add probe_model_path to config |

---

### Task 1: Emotion Probe Model

**Files:**
- Create: `music_brain/jepa/emotion_probe.py`
- Test: `tests/unit/test_emotion_probe.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/test_emotion_probe.py`:

```python
"""Tests for EmotionProbe model."""

import torch
import pytest
from music_brain.jepa.emotion_probe import EmotionProbe


class TestEmotionProbe:
    def test_output_shape(self):
        probe = EmotionProbe(latent_dim=256, hidden_dim=128)
        x = torch.randn(4, 256)  # batch of 4 pooled latents
        out = probe(x)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_output_range(self):
        probe = EmotionProbe(latent_dim=256, hidden_dim=128)
        x = torch.randn(32, 256)
        out = probe(x)
        assert out.min() >= -1.0, "Output below -1"
        assert out.max() <= 1.0, "Output above 1"

    def test_gradient_flows(self):
        probe = EmotionProbe(latent_dim=256, hidden_dim=128)
        x = torch.randn(4, 256)
        target = torch.tensor([[0.5, 0.3], [-0.2, 0.8], [0.0, 0.0], [0.9, -0.5]])
        out = probe(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        for p in probe.parameters():
            assert p.grad is not None, "Gradient not flowing"
            assert p.grad.abs().sum() > 0, "Zero gradient"

    def test_deterministic_eval(self):
        probe = EmotionProbe(latent_dim=256, hidden_dim=128)
        probe.eval()
        x = torch.randn(2, 256)
        out1 = probe(x)
        out2 = probe(x)
        assert torch.allclose(out1, out2), "Non-deterministic in eval mode"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv-coreml/bin/activate && python3 -m pytest tests/unit/test_emotion_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'music_brain.jepa.emotion_probe'`

- [ ] **Step 3: Implement EmotionProbe**

Create `music_brain/jepa/emotion_probe.py`:

```python
"""Emotion probe: small MLP that maps pooled JEPA latents to valence/arousal."""

from __future__ import annotations

import torch
import torch.nn as nn


class EmotionProbe(nn.Module):
    """MLP probe: pooled latent (256,) → (valence, arousal) in [-1, 1].

    Designed to be trained on frozen JEPA encoder embeddings.
    Exported to ONNX as a separate model for the C++ plugin.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, latent_dim) pooled latent vectors.
        Returns: (B, 2) with columns [valence, arousal] in [-1, 1]."""
        return self.mlp(x)
```

- [ ] **Step 4: Run tests**

Run: `source .venv-coreml/bin/activate && python3 -m pytest tests/unit/test_emotion_probe.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add music_brain/jepa/emotion_probe.py tests/unit/test_emotion_probe.py
git commit -m "feat: add EmotionProbe model — MLP latent-to-valence/arousal"
```

---

### Task 2: Build Emotion Dataset Manifest

**Files:**
- Create: `scripts/build_emotion_manifest.py`

This script scans DEAM and PMEmo dataset directories (assumed at `~/Datasets/`) and produces a manifest JSON consumable by `UnifiedEmotionDataset`.

- [ ] **Step 1: Write the manifest builder**

Create `scripts/build_emotion_manifest.py`:

```python
#!/usr/bin/env python3
"""
Build a unified emotion manifest JSON from DEAM and PMEmo datasets.

DEAM: ~/Datasets/by_domain/emotions/DEAM/
  - audio/ (*.mp3 or *.wav)
  - annotations/arousal.csv, annotations/valence.csv
  (mean-per-track annotations, columns: song_id, mean_valence/arousal)

PMEmo: ~/Datasets/by_domain/emotions/PMEmo/
  - audio/ (*.mp3 or *.wav)
  - annotations/static_annotations.csv
  (columns: musicId, mean_valence, mean_arousal — normalized 1-9 scale)

Output: data/emotion_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List


def scan_deam(root: Path) -> List[Dict]:
    """Parse DEAM dataset into manifest entries."""
    entries = []
    annotations_dir = root / "annotations"
    audio_dir = root / "audio"

    if not annotations_dir.exists() or not audio_dir.exists():
        print(f"DEAM not found at {root}")
        return entries

    # Load valence annotations
    valence_map: Dict[str, float] = {}
    valence_file = annotations_dir / "valence.csv"
    if valence_file.exists():
        with open(valence_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("song_id", row.get("musicId", "")).strip()
                val = row.get("mean_valence", row.get("valence_mean", ""))
                if sid and val:
                    valence_map[sid] = float(val)

    # Load arousal annotations
    arousal_map: Dict[str, float] = {}
    arousal_file = annotations_dir / "arousal.csv"
    if arousal_file.exists():
        with open(arousal_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("song_id", row.get("musicId", "")).strip()
                val = row.get("mean_arousal", row.get("arousal_mean", ""))
                if sid and val:
                    arousal_map[sid] = float(val)

    # Match audio files to annotations
    for audio_file in sorted(audio_dir.iterdir()):
        if audio_file.suffix.lower() not in (".wav", ".mp3", ".flac"):
            continue
        sid = audio_file.stem
        if sid in valence_map and sid in arousal_map:
            # DEAM uses 1-9 scale; normalize to [-1, 1]
            v_raw = valence_map[sid]
            a_raw = arousal_map[sid]
            valence = (v_raw - 5.0) / 4.0  # [1,9] → [-1,1]
            arousal = (a_raw - 5.0) / 4.0
            entries.append({
                "id": f"deam_{sid}",
                "dataset": "DEAM",
                "audio_path": str(audio_file),
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
            })

    print(f"DEAM: {len(entries)} tracks with annotations")
    return entries


def scan_pmemo(root: Path) -> List[Dict]:
    """Parse PMEmo dataset into manifest entries."""
    entries = []
    audio_dir = root / "audio"
    annotations_file = root / "annotations" / "static_annotations.csv"

    if not annotations_file.exists():
        print(f"PMEmo not found at {root}")
        return entries

    # Load annotations
    va_map: Dict[str, tuple] = {}
    with open(annotations_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("musicId", "").strip()
            v = row.get("mean_valence", row.get("valence_mean", ""))
            a = row.get("mean_arousal", row.get("arousal_mean", ""))
            if sid and v and a:
                va_map[sid] = (float(v), float(a))

    if audio_dir.exists():
        for audio_file in sorted(audio_dir.iterdir()):
            if audio_file.suffix.lower() not in (".wav", ".mp3", ".flac"):
                continue
            sid = audio_file.stem
            if sid in va_map:
                v_raw, a_raw = va_map[sid]
                valence = (v_raw - 5.0) / 4.0
                arousal = (a_raw - 5.0) / 4.0
                entries.append({
                    "id": f"pmemo_{sid}",
                    "dataset": "PMEmo",
                    "audio_path": str(audio_file),
                    "valence": round(valence, 4),
                    "arousal": round(arousal, 4),
                })

    print(f"PMEmo: {len(entries)} tracks with annotations")
    return entries


def assign_splits(entries: List[Dict], train: float = 0.7, val: float = 0.15) -> None:
    """Assign train/val/test splits deterministically by sorted ID."""
    entries.sort(key=lambda e: e["id"])
    n = len(entries)
    n_train = int(n * train)
    n_val = int(n * val)
    for i, entry in enumerate(entries):
        if i < n_train:
            entry["split"] = "train"
        elif i < n_train + n_val:
            entry["split"] = "val"
        else:
            entry["split"] = "test"


def main():
    parser = argparse.ArgumentParser(description="Build emotion dataset manifest")
    parser.add_argument("--datasets-root", default=os.path.expanduser("~/Datasets"),
                        help="Root datasets directory")
    parser.add_argument("--output", default="data/emotion_manifest.json",
                        help="Output manifest path")
    args = parser.parse_args()

    root = Path(args.datasets_root)
    entries: List[Dict] = []

    entries.extend(scan_deam(root / "by_domain" / "emotions" / "DEAM"))
    entries.extend(scan_pmemo(root / "by_domain" / "emotions" / "PMEmo"))

    if not entries:
        print("No entries found. Check dataset paths.")
        print(f"  Expected DEAM at: {root / 'by_domain' / 'emotions' / 'DEAM'}")
        print(f"  Expected PMEmo at: {root / 'by_domain' / 'emotions' / 'PMEmo'}")
        return

    assign_splits(entries)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2)

    n_train = sum(1 for e in entries if e["split"] == "train")
    n_val = sum(1 for e in entries if e["split"] == "val")
    n_test = sum(1 for e in entries if e["split"] == "test")
    print(f"\nManifest: {len(entries)} entries → {args.output}")
    print(f"  train={n_train}, val={n_val}, test={n_test}")
    print(f"  valence range: [{min(e['valence'] for e in entries):.2f}, "
          f"{max(e['valence'] for e in entries):.2f}]")
    print(f"  arousal range: [{min(e['arousal'] for e in entries):.2f}, "
          f"{max(e['arousal'] for e in entries):.2f}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/build_emotion_manifest.py
git commit -m "feat: add build_emotion_manifest.py for DEAM/PMEmo datasets"
```

---

### Task 3: Extract and Cache JEPA Embeddings

**Files:**
- Create: `scripts/extract_jepa_embeddings.py`

Runs the frozen JEPA encoder on all audio in the manifest, saves pooled latents + labels to a single `.pt` file for fast probe training.

- [ ] **Step 1: Write the extraction script**

Create `scripts/extract_jepa_embeddings.py`:

```python
#!/usr/bin/env python3
"""
Extract JEPA latent embeddings for all audio in an emotion manifest.

Saves a .pt file with:
  - embeddings: (N, 256) pooled latent vectors
  - valence: (N,) ground-truth valence
  - arousal: (N,) ground-truth arousal
  - ids: list of track IDs
  - splits: list of split assignments

Usage:
    python scripts/extract_jepa_embeddings.py
    python scripts/extract_jepa_embeddings.py --manifest data/emotion_manifest.json
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import logging

import librosa
import numpy as np
import torch

from music_brain.jepa.audio_jepa import AudioJEPAEncoder
from music_brain.jepa.config import AudioJEPAConfig
from music_brain.penta_core.ml.unified_emotion import UnifiedEmotionDataset

logger = logging.getLogger(__name__)


def load_encoder(checkpoint_path: str, device: torch.device) -> AudioJEPAEncoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = AudioJEPAConfig(**ckpt["config"])
    encoder = AudioJEPAEncoder(config=config)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval().to(device)
    return encoder


def audio_to_mel_tensor(
    audio: np.ndarray, sr: int = 22050, n_mels: int = 128,
    hop_length: int = 512, max_frames: int = 512,
) -> torch.Tensor:
    """Convert raw audio to mel tensor (1, 1, 128, 512)."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Pad or truncate to max_frames
    if mel_db.shape[1] < max_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_frames - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :max_frames]

    return torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).float()


def main():
    parser = argparse.ArgumentParser(description="Extract JEPA embeddings")
    parser.add_argument("--manifest", default="data/emotion_manifest.json")
    parser.add_argument("--checkpoint", default="checkpoints/audio_jepa/best_model.pt")
    parser.add_argument("--output", default="data/jepa_emotion_embeddings.pt")
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    encoder = load_encoder(args.checkpoint, device)
    logger.info("Encoder loaded from %s", args.checkpoint)

    # Load manifest
    import json
    with open(args.manifest) as f:
        manifest = json.load(f)
    logger.info("Manifest: %d entries", len(manifest))

    embeddings = []
    valences = []
    arousals = []
    ids = []
    splits = []
    errors = 0

    for i, entry in enumerate(manifest):
        try:
            audio, _ = librosa.load(entry["audio_path"], sr=args.sr, mono=True)
            mel = audio_to_mel_tensor(audio, sr=args.sr).to(device)

            with torch.no_grad():
                latent = encoder(mel)  # (1, T, 256)
                pooled = latent.mean(dim=1).cpu()  # (1, 256)

            embeddings.append(pooled)
            valences.append(entry["valence"])
            arousals.append(entry["arousal"])
            ids.append(entry["id"])
            splits.append(entry.get("split", "train"))

            if (i + 1) % 100 == 0:
                logger.info("Processed %d / %d", i + 1, len(manifest))

        except Exception as e:
            logger.warning("Skipping %s: %s", entry.get("id", "?"), e)
            errors += 1

    result = {
        "embeddings": torch.cat(embeddings, dim=0),  # (N, 256)
        "valence": torch.tensor(valences, dtype=torch.float32),
        "arousal": torch.tensor(arousals, dtype=torch.float32),
        "ids": ids,
        "splits": splits,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    logger.info("Saved %d embeddings to %s (%d errors)",
                len(embeddings), args.output, errors)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/extract_jepa_embeddings.py
git commit -m "feat: add extract_jepa_embeddings.py — cache JEPA latents for probe training"
```

---

### Task 4: Train Emotion Probe

**Files:**
- Create: `scripts/train_emotion_probe.py`

Trains the EmotionProbe on cached embeddings. Reads the `.pt` file from Task 3.

- [ ] **Step 1: Write training script**

Create `scripts/train_emotion_probe.py`:

```python
#!/usr/bin/env python3
"""
Train EmotionProbe on cached JEPA embeddings.

Usage:
    python scripts/train_emotion_probe.py
    python scripts/train_emotion_probe.py --embeddings data/jepa_emotion_embeddings.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from music_brain.jepa.emotion_probe import EmotionProbe

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train emotion probe")
    parser.add_argument("--embeddings", default="data/jepa_emotion_embeddings.pt")
    parser.add_argument("--output", default="checkpoints/emotion_probe/best_probe.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load cached embeddings
    data = torch.load(args.embeddings, weights_only=False)
    embeddings = data["embeddings"]  # (N, 256)
    valence = data["valence"]        # (N,)
    arousal = data["arousal"]        # (N,)
    splits = data["splits"]          # list of str

    targets = torch.stack([valence, arousal], dim=1)  # (N, 2)

    # Split by cached split assignments
    train_mask = torch.tensor([s == "train" for s in splits])
    val_mask = torch.tensor([s == "val" for s in splits])
    test_mask = torch.tensor([s == "test" for s in splits])

    train_ds = TensorDataset(embeddings[train_mask], targets[train_mask])
    val_ds = TensorDataset(embeddings[val_mask], targets[val_mask])
    test_ds = TensorDataset(embeddings[test_mask], targets[test_mask])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    logger.info("Train: %d, Val: %d, Test: %d", len(train_ds), len(val_ds), len(test_ds))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    probe = EmotionProbe(latent_dim=256, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        probe.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pred = probe(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # Validate
        probe.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = probe(x)
                val_loss += F.mse_loss(pred, y).item() * x.size(0)
        val_loss /= len(val_ds)

        logger.info("Epoch %d | train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "probe": probe.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "latent_dim": 256,
                "hidden_dim": 128,
            }, args.output)
            logger.info("  Saved best probe (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Test
    probe.load_state_dict(torch.load(args.output, weights_only=False)["probe"])
    probe.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred = probe(x)
            test_loss += F.mse_loss(pred, y).item() * x.size(0)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    test_loss /= len(test_ds)

    preds = torch.cat(all_preds)
    tgts = torch.cat(all_targets)
    corr_v = torch.corrcoef(torch.stack([preds[:, 0], tgts[:, 0]]))[0, 1]
    corr_a = torch.corrcoef(torch.stack([preds[:, 1], tgts[:, 1]]))[0, 1]

    print(f"\n=== Test Results ===")
    print(f"Test MSE:            {test_loss:.4f}")
    print(f"Valence correlation: {corr_v:.3f}")
    print(f"Arousal correlation: {corr_a:.3f}")
    print(f"Best val loss:       {best_val_loss:.4f}")
    print(f"Checkpoint:          {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_emotion_probe.py
git commit -m "feat: add train_emotion_probe.py — train probe on cached JEPA embeddings"
```

---

### Task 5: Export Emotion Probe to ONNX

**Files:**
- Create: `scripts/export_emotion_probe.py`

- [ ] **Step 1: Write export script**

Create `scripts/export_emotion_probe.py`:

```python
#!/usr/bin/env python3
"""
Export trained EmotionProbe to ONNX.

Usage:
    python scripts/export_emotion_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import logging

import numpy as np
import torch

from music_brain.jepa.emotion_probe import EmotionProbe

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export emotion probe to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/emotion_probe/best_probe.pt")
    parser.add_argument("--output", default="models/emotion_probe_v01.onnx")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    probe = EmotionProbe(
        latent_dim=ckpt.get("latent_dim", 256),
        hidden_dim=ckpt.get("hidden_dim", 128),
    )
    probe.load_state_dict(ckpt["probe"])
    probe.eval()

    # Export
    dummy = torch.randn(1, 256)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        probe, dummy, args.output,
        opset_version=17,
        input_names=["latent"],
        output_names=["emotion"],
    )
    logger.info("ONNX exported: %s", args.output)

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output)
    ort_out = sess.run(None, {"latent": dummy.numpy()})[0]
    with torch.no_grad():
        pt_out = probe(dummy).numpy()
    max_diff = np.abs(pt_out - ort_out).max()
    logger.info("Verification: max_diff=%.2e %s", max_diff,
                "PASS" if max_diff < 1e-5 else "FAIL")

    import os
    size_kb = os.path.getsize(args.output) / 1024
    logger.info("Model size: %.1f KB", size_kb)

    print(f"\n=== Export Summary ===")
    print(f"ONNX:     {args.output} ({size_kb:.0f} KB)")
    print(f"Input:    latent (1, 256)")
    print(f"Output:   emotion (1, 2) [valence, arousal]")
    print(f"Verified: max_diff={max_diff:.2e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/export_emotion_probe.py
git commit -m "feat: add export_emotion_probe.py — probe to ONNX for C++ plugin"
```

---

### Task 6: Wire Probe into AudioEmotionRunner (C++)

**Files:**
- Modify: `include/penta/ml/AudioEmotionRunner.h:45-52`
- Modify: `src/ml/AudioEmotionRunner.cpp:112-153,204-231`

- [ ] **Step 1: Add probe_model_path to config**

In `include/penta/ml/AudioEmotionRunner.h`, add to `AudioEmotionRunnerConfig`:

```cpp
struct AudioEmotionRunnerConfig {
    std::string model_path;                // Path to JEPA .onnx file
    std::string probe_model_path;          // Path to emotion probe .onnx file
    size_t sample_rate         = 48000;
    size_t ring_capacity       = 524288;
    float  slew_time_ms        = 20.0f;
    float  watchdog_timeout_ms = 100.0f;
    float  confidence_threshold = 0.3f;
};
```

- [ ] **Step 2: Add probe ONNX session to impl**

In `src/ml/AudioEmotionRunner.cpp`, in the `AudioEmotionRunnerImpl` struct, add after the `onnx` member:

```cpp
#ifdef ENABLE_ONNX_RUNTIME
    std::unique_ptr<penta::ml::ONNXInference> onnx;       // JEPA encoder
    std::unique_ptr<penta::ml::ONNXInference> probeOnnx;  // Emotion probe
#endif
```

In the `initialize` method, after loading the JEPA model, add:

```cpp
#ifdef ENABLE_ONNX_RUNTIME
    if (!config.probe_model_path.empty()) {
        probeOnnx = std::make_unique<penta::ml::ONNXInference>();
        if (!probeOnnx->loadModel(config.probe_model_path)) {
            probeOnnx.reset();
        }
    }
#endif
```

- [ ] **Step 3: Replace hardcoded mapping with probe inference**

In the `workerLoop()`, replace the emotion mapping section (after mean-pool latent) with:

```cpp
            // 4. Map to emotion: use probe ONNX if available, else hardcoded
            EmotionRunnerResult result;
#ifdef ENABLE_ONNX_RUNTIME
            if (probeOnnx && probeOnnx->isLoaded()) {
                // Run probe: (1, 256) → (1, 2) [valence, arousal]
                float probeOut[2] = {0.0f, 0.0f};
                probeOnnx->infer(pooledLatent.data(), probeOut);
                result.emotion.valence   = std::clamp(probeOut[0], -1.0f, 1.0f);
                result.emotion.arousal   = std::clamp((probeOut[1] + 1.0f) * 0.5f, 0.0f, 1.0f);
                result.emotion.dominance = std::clamp(
                    0.5f + 0.3f * result.emotion.arousal + 0.2f * std::abs(result.emotion.valence),
                    0.0f, 1.0f);
                result.emotion.confidence = 0.8f;  // Trained probe — fixed high confidence
            } else {
                result.emotion = mapLatentToEmotion(pooledLatent.data(), latentDim);
            }
#else
            result.emotion = mapLatentToEmotion(pooledLatent.data(), latentDim);
#endif
            result.dsp = mapEmotionToDSP(result.emotion);
```

- [ ] **Step 4: Set probe path in PluginProcessor**

In `src/plugin/PluginProcessor.cpp`, after the JEPA model path resolution, add:

```cpp
    // Resolve emotion probe model path
    auto probeFile = pluginFile.getChildFile(
        "Contents/Resources/models/emotion_probe_v01.onnx");
    if (!probeFile.existsAsFile())
      probeFile = pluginFile.getParentDirectory().getChildFile(
          "models/emotion_probe_v01.onnx");
    if (!probeFile.existsAsFile())
      probeFile = juce::File(
          "/Users/seanburdges/Dev/KmiDi/models/emotion_probe_v01.onnx");
    emotionConfig.probe_model_path = probeFile.getFullPathName().toStdString();
```

- [ ] **Step 5: Rebuild and verify**

```bash
cmake --build build-demo --target KellyPlugin_AU -j8 2>&1 | tail -5
```

Expected: builds without errors.

- [ ] **Step 6: Commit**

```bash
git add include/penta/ml/AudioEmotionRunner.h src/ml/AudioEmotionRunner.cpp src/plugin/PluginProcessor.cpp
git commit -m "feat: wire emotion probe ONNX into AudioEmotionRunner"
```

---

### Task 7: Add Emotion Auxiliary Loss to JEPA Trainer

**Files:**
- Modify: `music_brain/jepa/trainer.py:76-196`
- Modify: `config/jepa_training.yaml`

- [ ] **Step 1: Add emotion_loss_weight to training config**

In `config/jepa_training.yaml`, add under `training:`:

```yaml
training:
  epochs: 120
  batch_size: 12
  learning_rate: 3.0e-4
  weight_decay: 0.01
  warmup_epochs: 10
  save_every: 5
  early_stop_patience: 15
  mixed_precision: true
  gradient_clip: 1.0
  emotion_loss_weight: 0.1
  emotion_probe_checkpoint: "checkpoints/emotion_probe/best_probe.pt"
```

- [ ] **Step 2: Update TrainingConfig dataclass**

In `music_brain/jepa/config.py`, add to `TrainingConfig`:

```python
    emotion_loss_weight: float = 0.0
    emotion_probe_checkpoint: str = ""
```

- [ ] **Step 3: Add emotion auxiliary loss to train_audio_jepa**

In `music_brain/jepa/trainer.py`, modify `train_audio_jepa()` to optionally load a frozen probe and add emotion loss:

After the optimizer creation (line ~100), add:

```python
    # Optional emotion auxiliary loss
    emotion_probe = None
    emotion_dataloader = None
    if training.emotion_loss_weight > 0 and training.emotion_probe_checkpoint:
        from music_brain.jepa.emotion_probe import EmotionProbe
        probe_ckpt = torch.load(training.emotion_probe_checkpoint,
                                map_location="cpu", weights_only=False)
        emotion_probe = EmotionProbe(
            latent_dim=probe_ckpt.get("latent_dim", 256),
            hidden_dim=probe_ckpt.get("hidden_dim", 128),
        )
        emotion_probe.load_state_dict(probe_ckpt["probe"])
        emotion_probe.to(device).eval()
        for p in emotion_probe.parameters():
            p.requires_grad = False
        logger.info("Emotion auxiliary loss enabled (weight=%.2f)", training.emotion_loss_weight)
```

Then in the training loop, after the main JEPA loss (line 128), add:

```python
                loss = F.mse_loss(pred, z_target)

                # Emotion auxiliary: encourage encoder to produce emotion-relevant latents
                if emotion_probe is not None:
                    z_pooled = z.mean(dim=1)  # (B, latent_dim)
                    emotion_pred = emotion_probe(z_pooled)
                    # Use probe's own output as soft target (self-distillation)
                    # This steers latent geometry without requiring labels in main dataloader
                    with torch.no_grad():
                        z_target_pooled = z_target.mean(dim=1)
                        emotion_target = emotion_probe(z_target_pooled)
                    emotion_loss = F.mse_loss(emotion_pred, emotion_target)
                    loss = loss + training.emotion_loss_weight * emotion_loss
```

- [ ] **Step 4: Commit**

```bash
git add music_brain/jepa/trainer.py music_brain/jepa/config.py config/jepa_training.yaml
git commit -m "feat: add emotion auxiliary loss to JEPA trainer"
```

---

### Task 8: Update Demo Script

**Files:**
- Modify: `scripts/demo_proof_of_life.py`

- [ ] **Step 1: Add probe inference to demo**

In `scripts/demo_proof_of_life.py`, replace the `latent_to_emotion()` heuristic function with:

```python
def load_emotion_probe(checkpoint_path: str = "checkpoints/emotion_probe/best_probe.pt"):
    """Load trained emotion probe if available."""
    from pathlib import Path
    if not Path(checkpoint_path).exists():
        return None
    from music_brain.jepa.emotion_probe import EmotionProbe
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    probe = EmotionProbe(
        latent_dim=ckpt.get("latent_dim", 256),
        hidden_dim=ckpt.get("hidden_dim", 128),
    )
    probe.load_state_dict(ckpt["probe"])
    probe.eval()
    return probe


def latent_to_emotion(latent: np.ndarray, probe=None) -> dict:
    """Map latent to emotion using trained probe or heuristic fallback."""
    pooled = latent[0].mean(axis=0)  # (256,)

    if probe is not None:
        with torch.no_grad():
            inp = torch.from_numpy(pooled).unsqueeze(0).float()
            out = probe(inp).squeeze(0).numpy()
        valence = float(out[0])
        arousal = float((out[1] + 1.0) * 0.5)  # [-1,1] → [0,1]
    else:
        # Heuristic fallback
        mean_val = float(pooled.mean())
        std_val = float(pooled.std())
        energy = float(np.abs(pooled).mean())
        skew = float(np.mean((pooled - mean_val) ** 3) / (std_val ** 3 + 1e-8))
        valence = float(np.tanh(skew * 2))
        arousal = float(np.clip(energy * 3, 0, 1))

    dominance = float(np.clip(0.5 + 0.3 * arousal + 0.2 * abs(valence), 0, 1))
    confidence = 0.8 if probe else 0.3

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "dominance": round(dominance, 3),
        "confidence": round(confidence, 3),
    }
```

Update `main()` to load the probe and pass it:

```python
    # In main(), after setting up inference backend:
    probe = load_emotion_probe()
    if probe:
        print("Emotion probe: TRAINED (using learned mapping)")
    else:
        print("Emotion probe: not found (using heuristic fallback)")

    # In the window loop, change the call:
    emotion = latent_to_emotion(latent, probe=probe)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/demo_proof_of_life.py
git commit -m "feat: demo script uses trained emotion probe when available"
```
