"""
Audio-JEPA: Joint-Embedding Predictive Architecture for audio.

Encodes mel-spectrograms into latent representations and predicts
masked regions in latent space — no pixel-level reconstruction needed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from music_brain.jepa.config import AudioJEPAConfig

_TIER_PARAMS = {
    "small": {"channels": [16, 32, 64, 128], "latent_dim": 128},
    "medium": {"channels": [32, 64, 128, 256], "latent_dim": 256},
    "large": {"channels": [64, 128, 256, 512], "latent_dim": 512},
}


class AudioJEPAEncoder(nn.Module):
    """
    Convolutional encoder that maps mel-spectrograms to a sequence of
    latent vectors.

    Input:  ``(B, 1, n_mels, T)``
    Output: ``(B, T', latent_dim)``  where ``T' = T // 4``
    """

    def __init__(
        self,
        tier: str = "medium",
        latent_dim: Optional[int] = None,
        n_mels: int = 128,
        config: Optional[AudioJEPAConfig] = None,
    ):
        super().__init__()
        if config is not None:
            tier = config.tier
            latent_dim = config.latent_dim
            n_mels = config.n_mels

        params = _TIER_PARAMS.get(tier, _TIER_PARAMS["medium"])
        ch = params["channels"]
        self.latent_dim = latent_dim or params["latent_dim"]

        self.conv = nn.Sequential(
            nn.Conv2d(1, ch[0], 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(ch[0]),
            nn.GELU(),
            nn.Conv2d(ch[0], ch[1], 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(ch[1]),
            nn.GELU(),
            nn.Conv2d(ch[1], ch[2], 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(ch[2]),
            nn.GELU(),
            nn.Conv2d(ch[2], ch[3], 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(ch[3]),
            nn.GELU(),
        )

        freq_out = n_mels // 16
        self.proj = nn.Linear(ch[3] * freq_out, self.latent_dim)
        self.layer_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, 1, n_mels, T)`` mel-spectrogram
        Returns:
            ``(B, T, latent_dim)`` latent sequence
        """
        h = self.conv(x)               # (B, C, freq_out, T)
        B, C, F_, T = h.shape
        h = h.permute(0, 3, 1, 2)      # (B, T, C, F_)
        h = h.reshape(B, T, C * F_)    # (B, T, C*F_)
        h = self.proj(h)               # (B, T, latent_dim)
        return self.layer_norm(h)


class LatentPredictor(nn.Module):
    """
    Lightweight Transformer predictor that reconstructs masked latent
    positions from their surrounding context.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_masked: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_masked: ``(B, T, D)`` latent sequence with masked positions zeroed.
        Returns:
            ``(B, T, D)`` predicted latent sequence.
        """
        h = self.transformer(z_masked)
        return self.out_proj(h)


class EMATargetEncoder(nn.Module):
    """
    Exponential-moving-average copy of the encoder used as the
    prediction target (teacher) in JEPA training.
    """

    def __init__(self, encoder: AudioJEPAEncoder, momentum: float = 0.996):
        super().__init__()
        self.encoder = self._copy(encoder)
        self.momentum = momentum
        for p in self.encoder.parameters():
            p.requires_grad = False

    @staticmethod
    def _copy(model: nn.Module) -> nn.Module:
        import copy
        return copy.deepcopy(model)

    @torch.no_grad()
    def update(self, online_encoder: AudioJEPAEncoder) -> None:
        for p_ema, p_online in zip(
            self.encoder.parameters(), online_encoder.parameters()
        ):
            p_ema.data.mul_(self.momentum).add_(
                p_online.data, alpha=1.0 - self.momentum
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
