#!/usr/bin/env python3
"""
Extract and cache JEPA embeddings for emotion probe training.

Reads a manifest JSON (from build_emotion_manifest.py), runs each audio file
through the frozen JEPA encoder, mean-pools over the time dimension, and saves
all embeddings plus targets to a single .pt file.

Preprocessing pipeline per file:
  librosa.load(sr=22050, mono=True) → mel spectrogram (n_mels=128, hop=512)
  → normalize to [0, 1] → pad/truncate to 512 frames
  → shape (1, 1, 128, 512) → encoder → mean-pool time dim → (256,)

Usage:
    python scripts/extract_jepa_embeddings.py
    python scripts/extract_jepa_embeddings.py \\
        --manifest data/emotion_manifest.json \\
        --checkpoint checkpoints/audio_jepa/best_model.pt \\
        --output data/jepa_emotion_embeddings.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from music_brain.jepa.audio_jepa import AudioJEPAEncoder
from music_brain.jepa.config import AudioJEPAConfig

logger = logging.getLogger(__name__)

# Fixed preprocessing parameters (must match encoder training)
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
MAX_FRAMES = 512
LATENT_DIM = 256


def get_device() -> torch.device:
    """Select MPS if available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_encoder(checkpoint_path: Path, device: torch.device) -> AudioJEPAEncoder:
    """Load frozen JEPA encoder from checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = AudioJEPAConfig(**ckpt["config"])
    encoder = AudioJEPAEncoder(config=config)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    logger.info(
        "Loaded encoder: tier=%s, latent_dim=%d, epoch=%d, loss=%.6f",
        config.tier, config.latent_dim, ckpt["epoch"], ckpt["loss"],
    )
    return encoder


def audio_to_mel(audio_path: str, device: torch.device) -> torch.Tensor:
    """Load audio and compute mel spectrogram, padded/truncated to MAX_FRAMES.

    Normalises to [0, 1] before returning.

    Returns:
        Tensor of shape (1, 1, N_MELS, MAX_FRAMES) on the given device.
    """
    import librosa
    import numpy as np

    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    # Convert to dB, then normalise to [0, 1]
    mel_db = librosa.power_to_db(mel, ref=np.max)   # shape: (N_MELS, T)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    denom = mel_max - mel_min if mel_max > mel_min else 1.0
    mel_norm = (mel_db - mel_min) / denom            # [0, 1]

    # Pad or truncate to MAX_FRAMES
    n_frames = mel_norm.shape[1]
    if n_frames < MAX_FRAMES:
        pad = MAX_FRAMES - n_frames
        mel_norm = np.pad(mel_norm, ((0, 0), (0, pad)), mode="constant")
    else:
        mel_norm = mel_norm[:, :MAX_FRAMES]

    # (1, 1, N_MELS, MAX_FRAMES)
    tensor = torch.from_numpy(mel_norm).float().unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


@torch.no_grad()
def encode_file(audio_path: str, encoder: AudioJEPAEncoder, device: torch.device) -> torch.Tensor:
    """Encode a single audio file to a (LATENT_DIM,) embedding via mean-pooling."""
    mel = audio_to_mel(audio_path, device)            # (1, 1, N_MELS, MAX_FRAMES)
    latent = encoder(mel)                              # (1, T, latent_dim)
    embedding = latent.mean(dim=1).squeeze(0)          # (latent_dim,)
    return embedding.cpu()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract and cache JEPA embeddings for emotion probe training."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/emotion_manifest.json"),
        help="Emotion manifest JSON (default: data/emotion_manifest.json)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/audio_jepa/best_model.pt"),
        help="JEPA encoder checkpoint (default: checkpoints/audio_jepa/best_model.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/jepa_emotion_embeddings.pt"),
        help="Output .pt file (default: data/jepa_emotion_embeddings.pt)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve paths relative to project root when not absolute
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else Path(_PROJECT_ROOT) / args.manifest
    )
    checkpoint_path = (
        args.checkpoint if args.checkpoint.is_absolute() else Path(_PROJECT_ROOT) / args.checkpoint
    )
    output_path = (
        args.output if args.output.is_absolute() else Path(_PROJECT_ROOT) / args.output
    )

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    # Load manifest
    with manifest_path.open() as f:
        manifest = json.load(f)
    logger.info("Loaded manifest: %d entries", len(manifest))

    device = get_device()
    logger.info("Using device: %s", device)

    encoder = load_encoder(checkpoint_path, device)

    embeddings_list: list[torch.Tensor] = []
    valences: list[float] = []
    arousals: list[float] = []
    ids: list[str] = []
    splits: list[str] = []
    error_count = 0

    for i, entry in enumerate(manifest):
        if i > 0 and i % 100 == 0:
            logger.info("Progress: %d/%d (errors so far: %d)", i, len(manifest), error_count)

        try:
            emb = encode_file(entry["audio_path"], encoder, device)
            embeddings_list.append(emb)
            valences.append(float(entry["valence"]))
            arousals.append(float(entry["arousal"]))
            ids.append(entry["id"])
            splits.append(entry["split"])
        except Exception as exc:
            logger.warning("Error encoding %s: %s", entry.get("audio_path", "?"), exc)
            error_count += 1

    logger.info(
        "Encoding complete: %d successful, %d errors",
        len(embeddings_list), error_count,
    )

    if not embeddings_list:
        logger.error("No embeddings extracted — aborting.")
        return 1

    embeddings = torch.stack(embeddings_list)          # (N, latent_dim)
    valence_tensor = torch.tensor(valences)
    arousal_tensor = torch.tensor(arousals)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "valence": valence_tensor,
            "arousal": arousal_tensor,
            "ids": ids,
            "splits": splits,
        },
        str(output_path),
    )
    logger.info("Saved embeddings to %s  shape=%s", output_path, tuple(embeddings.shape))
    print(f"\nEmbeddings saved: {output_path}")
    print(f"  shape:  {tuple(embeddings.shape)}")
    print(f"  errors: {error_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
