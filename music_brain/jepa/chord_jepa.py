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
        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

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
