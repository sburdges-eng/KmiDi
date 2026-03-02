"""Control embedding for UMP aggregated state (mean, std, slope per CC)."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

# Match ump_aggregate.BINS (14-bit effective)
BINS = 16384


class UMPControlEmbedding:
    """Embeds aggregated control state (mean, std, slope) into a single vector.

    Use from JEPA context encoder to bias latent from control state.
    See docs/UMP_JEPA_EXPRESSIVE_CONDITIONING.md.
    """

    def __init__(self, bins: int = BINS, embed_dim: int = 64):
        self.bins = bins
        self.embed_dim = embed_dim
        self._emb = None
        if nn is not None:
            self._emb = nn.Embedding(bins, embed_dim)

    def forward(self, control_dict: dict[str, int]) -> "torch.Tensor":
        """Return (embed_dim,) for one window. Sum mean + std + slope embeddings."""
        if self._emb is None or torch is None:
            raise RuntimeError("PyTorch required for UMPControlEmbedding.forward")
        mean = self._emb(torch.tensor(control_dict["mean"], dtype=torch.long))
        std = self._emb(torch.tensor(control_dict["std"], dtype=torch.long))
        slope = self._emb(torch.tensor(control_dict["slope"], dtype=torch.long))
        return (mean + std + slope).squeeze(0)

    def __call__(self, control_dict: dict[str, int]) -> "torch.Tensor":
        return self.forward(control_dict)


if torch is not None and nn is not None:

    class UMPControlEmbeddingModule(nn.Module):
        """nn.Module version for use in config-driven training."""

        def __init__(self, bins: int = BINS, embed_dim: int = 64):
            super().__init__()
            self.embedding = nn.Embedding(bins, embed_dim)

        def forward(self, control_dict: dict[str, int]) -> torch.Tensor:
            device = next(self.parameters()).device
            mean = self.embedding(torch.tensor(control_dict["mean"], dtype=torch.long, device=device))
            std = self.embedding(torch.tensor(control_dict["std"], dtype=torch.long, device=device))
            slope = self.embedding(torch.tensor(control_dict["slope"], dtype=torch.long, device=device))
            return (mean + std + slope).squeeze(0)
