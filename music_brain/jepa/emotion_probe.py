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
