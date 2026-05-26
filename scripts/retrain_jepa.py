#!/usr/bin/env python3
"""
Retrain Audio JEPA with emotion auxiliary loss on emotion-labeled audio.

Uses AudioMelDataset to load emotion audio (RAVDESS/CREMA-D/TESS),
trains with MSE reconstruction + emotion auxiliary loss, saves checkpoints.

Usage:
    python scripts/retrain_jepa.py
    python scripts/retrain_jepa.py --epochs 50 --emotion-weight 0.1
    python scripts/retrain_jepa.py --resume checkpoints/audio_jepa/best_model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from music_brain.jepa.audio_jepa import (  # noqa: E402
    AudioJEPAEncoder,
    EMATargetEncoder,
    LatentPredictor,
)
from music_brain.jepa.config import AudioJEPAConfig  # noqa: E402
from music_brain.jepa.datasets import AudioMelDataset  # noqa: E402
from music_brain.jepa.masking import mask_latents  # noqa: E402

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_audio_files(manifest_path: str) -> list[str]:
    """Extract audio file paths from the emotion manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    paths = [e["audio_path"] for e in manifest if os.path.exists(e["audio_path"])]
    logger.info("Collected %d audio files from manifest", len(paths))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Retrain JEPA with emotion auxiliary loss")
    parser.add_argument(
        "--manifest",
        default="data/emotion_manifest.json",
        help="Emotion manifest JSON (for audio file paths)",
    )
    parser.add_argument(
        "--resume", default="checkpoints/audio_jepa/best_model.pt", help="Checkpoint to resume from"
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/audio_jepa_v2", help="Output checkpoint directory"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--emotion-weight", type=float, default=0.1, help="Weight for emotion auxiliary loss"
    )
    parser.add_argument(
        "--probe-checkpoint",
        default="checkpoints/emotion_probe/best_probe.pt",
        help="Trained emotion probe for auxiliary loss",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    device = get_device()
    logger.info("Device: %s", device)

    # Load existing checkpoint to get config
    logger.info("Loading checkpoint: %s", args.resume)
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    config = AudioJEPAConfig(**ckpt["config"])
    start_epoch = ckpt["epoch"] + 1
    logger.info(
        "Resuming from epoch %d (loss=%.6f), config: tier=%s latent_dim=%d",
        start_epoch,
        ckpt["loss"],
        config.tier,
        config.latent_dim,
    )

    # Build dataset from emotion manifest audio files
    audio_files = collect_audio_files(args.manifest)
    if not audio_files:
        logger.error("No audio files found. Run build_emotion_manifest.py first.")
        return

    dataset = AudioMelDataset(
        files=audio_files,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
        max_frames=config.max_frames,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    logger.info("Dataset: %d files, %d batches per epoch", len(dataset), len(dataloader))

    # Build models
    encoder = AudioJEPAEncoder(config=config).to(device)
    encoder.load_state_dict(ckpt["encoder"])

    predictor = LatentPredictor(latent_dim=config.latent_dim).to(device)
    if "predictor" in ckpt:
        predictor.load_state_dict(ckpt["predictor"])

    teacher = EMATargetEncoder(encoder)
    if "teacher" in ckpt:
        teacher.encoder.load_state_dict(ckpt["teacher"])

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            logger.warning("Could not restore optimizer state, starting fresh")

    # Load emotion probe for auxiliary loss
    emotion_probe = None
    if args.emotion_weight > 0 and Path(args.probe_checkpoint).exists():
        from music_brain.jepa.emotion_probe import EmotionProbe

        probe_ckpt = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=False)
        emotion_probe = EmotionProbe(
            latent_dim=probe_ckpt.get("latent_dim", 256),
            hidden_dim=probe_ckpt.get("hidden_dim", 128),
        )
        emotion_probe.load_state_dict(probe_ckpt["probe"])
        emotion_probe.to(device).eval()
        for p in emotion_probe.parameters():
            p.requires_grad = False
        logger.info("Emotion auxiliary loss enabled (weight=%.2f)", args.emotion_weight)
    else:
        logger.info("No emotion auxiliary loss (weight=0 or probe not found)")

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = ckpt["loss"]
    patience_counter = 0

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info(
        "Starting training: %d epochs, batch_size=%d, lr=%.1e",
        args.epochs,
        args.batch_size,
        args.lr,
    )
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        actual_epoch = start_epoch + epoch
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_emotion = 0.0
        n_batches = 0
        t0 = time.time()

        encoder.train()
        predictor.train()

        for batch in dataloader:
            batch = batch.to(device)

            with torch.amp.autocast(
                "cuda" if device.type == "cuda" else "cpu",
                enabled=use_amp,
            ):
                z = encoder(batch)
                with torch.no_grad():
                    z_target = teacher(batch)

                masked, _ = mask_latents(
                    z,
                    mask_ratio=config.mask_ratio,
                    block_size=config.mask_block_size,
                )
                pred = predictor(masked)
                recon_loss = F.mse_loss(pred, z_target)
                loss = recon_loss

                # Emotion auxiliary loss
                if emotion_probe is not None:
                    z_pooled = z.mean(dim=1)
                    emotion_pred = emotion_probe(z_pooled)
                    with torch.no_grad():
                        z_target_pooled = z_target.mean(dim=1)
                        emotion_target = emotion_probe(z_target_pooled)
                    emo_loss = F.mse_loss(emotion_pred, emotion_target)
                    loss = loss + args.emotion_weight * emo_loss
                    epoch_emotion += emo_loss.item()

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()),
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()),
                    1.0,
                )
                optimizer.step()

            teacher.update(encoder)
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_emotion = epoch_emotion / max(n_batches, 1) if emotion_probe else 0
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d (abs %d) | loss=%.6f recon=%.6f emo=%.6f | %.1fs",
            epoch,
            actual_epoch,
            avg_loss,
            avg_recon,
            avg_emotion,
            elapsed,
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "teacher": teacher.encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": actual_epoch,
                    "loss": avg_loss,
                    "config": asdict(config),
                },
                ckpt_path,
            )
            logger.info("  -> New best (loss=%.6f) saved to %s", avg_loss, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                break

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "epoch": actual_epoch,
                    "loss": avg_loss,
                    "config": asdict(config),
                },
                os.path.join(args.output_dir, f"epoch_{actual_epoch}.pt"),
            )

    logger.info("=" * 60)
    logger.info("Training complete. Best loss: %.6f", best_loss)
    logger.info("Checkpoint dir: %s", args.output_dir)
    print("\nNext steps:")
    print(
        f"  1. Copy best checkpoint: cp {args.output_dir}/best_model.pt checkpoints/audio_jepa/best_model.pt"  # noqa: E501
    )
    print(
        f"  2. Re-export ONNX:  python scripts/export_audio_jepa.py --checkpoint {args.output_dir}/best_model.pt"  # noqa: E501
    )
    print(
        f"  3. Re-extract embeddings: python scripts/extract_jepa_embeddings.py --checkpoint {args.output_dir}/best_model.pt"  # noqa: E501
    )
    print("  4. Retrain probe:  python scripts/train_emotion_probe.py")
    print("  5. Export probe:   python scripts/export_emotion_probe.py")


if __name__ == "__main__":
    main()
