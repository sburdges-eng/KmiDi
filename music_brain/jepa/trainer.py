"""
Unified JEPA training loop.

Supports Audio-JEPA and Chord-JEPA with SageMaker, local CUDA/MPS/CPU,
checkpointing, ONNX export, and experiment tracking.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from music_brain.jepa.audio_jepa import (
    AudioJEPAEncoder,
    EMATargetEncoder,
    LatentPredictor,
)
from music_brain.jepa.chord_jepa import ChordEmbedding, ChordJEPA
from music_brain.jepa.config import (
    AudioJEPAConfig,
    ChordJEPAConfig,
    TrainingConfig,
)
from music_brain.jepa.masking import mask_latents

logger = logging.getLogger(__name__)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    state: Dict,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    logger.info("Checkpoint saved: %s", path)


def export_onnx(
    model: nn.Module,
    out_path: str,
    dummy_input: torch.Tensor,
    input_names: list,
    output_names: list,
) -> None:
    model.cpu().eval()
    dummy_input = dummy_input.cpu()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
    )
    logger.info("ONNX export: %s", out_path)


def train_audio_jepa(
    dataloader: DataLoader,
    config: AudioJEPAConfig,
    training: TrainingConfig,
    checkpoint_dir: str = "checkpoints/audio_jepa",
    export_path: Optional[str] = None,
) -> Dict:
    """
    Full Audio-JEPA training loop with EMA teacher.

    Returns:
        Dict of final training metrics.
    """
    device = get_device()
    logger.info("Audio-JEPA training on %s (%d batches)", device, len(dataloader))

    encoder = AudioJEPAEncoder(config=config).to(device)
    predictor = LatentPredictor(latent_dim=config.latent_dim).to(device)
    teacher = EMATargetEncoder(encoder)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    use_amp = training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_loss = float("inf")
    patience_counter = 0
    metrics: Dict = {"epochs": [], "best_loss": float("inf")}

    for epoch in range(training.epochs):
        epoch_loss = 0.0
        n_batches = 0

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
                loss = F.mse_loss(pred, z_target)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if training.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()),
                        training.gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if training.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()),
                        training.gradient_clip,
                    )
                optimizer.step()

            teacher.update(encoder)
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        metrics["epochs"].append({"epoch": epoch, "loss": avg_loss})
        logger.info("Epoch %d | Loss %.6f", epoch, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_checkpoint(
                {
                    "encoder": encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "teacher": teacher.encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "config": asdict(config),
                },
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= training.early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if (epoch + 1) % training.save_every == 0:
            save_checkpoint(
                {
                    "encoder": encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                },
                os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"),
            )

    metrics["best_loss"] = best_loss

    if export_path:
        dummy = torch.randn(1, 1, config.n_mels, config.max_frames)
        export_onnx(encoder, export_path, dummy, ["mel"], ["latent"])

    return metrics


def train_chord_jepa(
    dataloader: DataLoader,
    config: ChordJEPAConfig,
    training: TrainingConfig,
    checkpoint_dir: str = "checkpoints/chord_jepa",
) -> Dict:
    """
    Full Chord-JEPA training loop.

    Returns:
        Dict of final training metrics.
    """
    device = get_device()
    logger.info("Chord-JEPA training on %s (%d batches)", device, len(dataloader))

    embedding = ChordEmbedding(
        num_chords=config.num_chords, d_model=config.d_model
    ).to(device)
    model = ChordJEPA(config=config).to(device)

    optimizer = torch.optim.AdamW(
        list(embedding.parameters()) + list(model.parameters()),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    best_loss = float("inf")
    metrics: Dict = {"epochs": [], "best_loss": float("inf")}

    for epoch in range(training.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for chords in dataloader:
            chords = chords.to(device)
            z = embedding(chords)
            masked, _ = mask_latents(
                z,
                mask_ratio=config.mask_ratio,
                block_size=config.mask_block_size,
            )
            pred = model(masked)
            loss = F.cross_entropy(
                pred.reshape(-1, config.num_chords),
                chords.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            if training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(embedding.parameters()) + list(model.parameters()),
                    training.gradient_clip,
                )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        metrics["epochs"].append({"epoch": epoch, "loss": avg_loss})
        logger.info("Epoch %d | Loss %.6f", epoch, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                {
                    "embedding": embedding.state_dict(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "config": asdict(config),
                },
                os.path.join(checkpoint_dir, "best_model.pt"),
            )

    metrics["best_loss"] = best_loss
    return metrics
