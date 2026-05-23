"""Latent-space regularizers: norm projection + variance floor.

Two small ``nn.Module`` wrappers that compose with the existing JEPA
encoders and the world-model rollout. Neither owns trainable
parameters; both are pure functional operators whose only state is the
configured threshold. That keeps them safe to slot anywhere — including
inside frozen / no-grad inference loops — without touching the host
model's optimizer.

  - ``L2NormProjection(radius)`` clamps the last-axis L2 norm to
    ``radius``. Inputs strictly inside the ball pass through unchanged
    (preserves direction *and* magnitude); inputs at-or-outside the
    ball are radially scaled back to the boundary. This is the
    canonical safety net against unbounded latent growth during long
    world-model rollouts (see ``music_brain/world_model.py`` —
    autoregressive GRU rollouts have no built-in bound).

  - ``VarianceFloor(min_var)`` is a *loss-only* regularizer. It returns
    ``(z_unchanged, aux_loss)`` where ``aux_loss`` is the summed
    deficit between ``min_var`` and the per-feature batch variance,
    rectified through ReLU. Plug ``aux_loss * λ`` into the training
    loss to prevent JEPA-style feature collapse without distorting the
    forward pass at inference.

Both are stable under autograd: ``L2NormProjection`` uses the standard
``clamp(min=1)`` trick so the subgradient at the boundary points
inward, and ``VarianceFloor`` uses torch's ``var`` and ``relu`` which
are everywhere subdifferentiable.
"""

from __future__ import annotations


def _torch():
    import torch
    return torch


class L2NormProjection:
    """Project the last-axis L2 norm of an input tensor to ``radius``.

    Inputs ``z`` of shape ``(..., D)`` whose row norms exceed ``radius``
    are radially scaled to lie on the sphere of that radius. Inputs
    inside the ball pass through. Zero vectors stay zero (no NaNs).
    """

    def __init__(self, radius: float = 1.0) -> None:
        if not (radius > 0.0):
            raise ValueError(f"radius must be positive, got {radius}")
        self.radius = float(radius)

    def __call__(self, z):
        torch = _torch()
        norms = torch.linalg.norm(z, dim=-1, keepdim=True)
        # clamp(min=radius) ensures we only ever scale *down*. The +eps
        # keeps the divide finite when the row is exactly zero.
        scale = self.radius / norms.clamp(min=self.radius).clamp_min(1e-12)
        return z * scale

    def forward(self, z):
        return self.__call__(z)


class VarianceFloor:
    """Aux-loss regularizer that penalizes per-feature variance below ``min_var``.

    Computes the per-feature batch variance across the leading dim,
    rectifies the deficit ``relu(min_var - var)``, and returns the sum
    as ``aux_loss``. Combine into the training loss as
    ``loss + λ * aux``.
    """

    def __init__(self, min_var: float = 1e-3) -> None:
        if not (min_var > 0.0):
            raise ValueError(f"min_var must be positive, got {min_var}")
        self.min_var = float(min_var)

    def __call__(self, z):
        torch = _torch()
        if z.dim() < 2:
            raise ValueError("input must have at least a batch dim and a feature dim")
        if z.shape[0] < 2:
            raise ValueError(
                "VarianceFloor requires batch size >= 2 to estimate variance")
        # unbiased=False matches the JEPA collapse-prevention literature.
        var = z.var(dim=0, unbiased=False)
        deficit = torch.relu(self.min_var - var)
        aux = deficit.sum()
        return z, aux

    def forward(self, z):
        return self.__call__(z)


__all__ = ["L2NormProjection", "VarianceFloor"]
