"""
Chord-JEPA: Self-supervised chord sequence understanding.

Learns chord progression representations via masked prediction
in latent space — captures harmonic relationships without
explicit labels.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from music_brain.jepa.config import ChordJEPAConfig

NUM_CHORDS = 170
CHORD_SEQ_LEN = 64


class ChordJEPA(nn.Module):
    """
    Transformer-based Chord-JEPA that embeds chord tokens and
    predicts masked positions.

    Input:  embedded chord sequence ``(B, T, d_model)``
    Output: per-position logits ``(B, T, num_chords)``
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = NUM_CHORDS,
        num_heads: int = 8,
        num_layers: int = 4,
        seq_len: int = CHORD_SEQ_LEN,
        dropout: float = 0.1,
        config: Optional[ChordJEPAConfig] = None,
    ):
        super().__init__()
        if config is not None:
            d_model = config.d_model
            num_classes = config.num_chords
            num_heads = config.num_heads
            num_layers = config.num_layers
            seq_len = config.seq_len
            dropout = config.dropout

        self.d_model = d_model
        self.num_classes = num_classes
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: ``(B, T, d_model)`` — embedded (and possibly masked) chord
               sequence.
        Returns:
            ``(B, T, num_chords)`` logits for each position.
        """
        T = z.size(1)
        z = z + self.pos_embed[:, :T, :]
        h = self.transformer(z)
        return self.head(h)


class ChordEmbedding(nn.Module):
    """Learnable embedding table for chord tokens."""

    def __init__(
        self,
        num_chords: int = NUM_CHORDS,
        d_model: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_chords, d_model)

    def forward(self, chord_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chord_ids: ``(B, T)`` integer chord indices.
        Returns:
            ``(B, T, d_model)`` embedded representations.
        """
        return self.embedding(chord_ids)


def encode_chord_to_frame(
    embedded_chords: torch.Tensor,
    *,
    audio_z: torch.Tensor,
    time_index: int,
    provenance,
    emotion_va=(0.0, 0.0),
    metadata=None,
):
    """Pack an embedded chord sequence + paired audio latents into a LatentFrame.

    The Chord-JEPA stack operates in ``d_model``-space (already embedded
    by ``ChordEmbedding``), so this helper packages the embedded
    sequence directly. ``audio_z`` is required because ``LatentFrame``
    treats audio as the primary modality; pass a zero placeholder of
    shape ``(1, D_audio)`` for chord-only flows.

    Args:
        embedded_chords: ``(1, T_c, d_model)`` or ``(T_c, d_model)``
            chord embeddings (single sample).
        audio_z: paired audio latents ``(T, D_audio)`` for the same
            window.
        time_index: monotonic chunk index.
        provenance: ``IntentProvenance``.
        emotion_va: optional pooled VA tag.
        metadata: diagnostics bag.
    """
    from music_brain.latent.latent_frame import LatentFrame  # noqa: PLC0415

    if embedded_chords.dim() == 3:
        if embedded_chords.shape[0] != 1:
            raise ValueError(
                "encode_chord_to_frame expects a single sample; "
                f"got batch dim {embedded_chords.shape[0]}"
            )
        chord_z = embedded_chords.squeeze(0).contiguous()
    elif embedded_chords.dim() == 2:
        chord_z = embedded_chords.contiguous()
    else:
        raise ValueError(
            f"embedded_chords must be 2-D or 3-D; got shape {tuple(embedded_chords.shape)}"
        )
    meta = {"source": "chord_jepa"}
    if metadata:
        meta.update(metadata)
    return LatentFrame(
        audio_z=audio_z,
        chord_z=chord_z,
        emotion_va=emotion_va,
        time_index=int(time_index),
        provenance=provenance,
        metadata=meta,
    )
