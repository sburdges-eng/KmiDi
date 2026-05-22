"""LoRA-style low-rank adapter for soft-prompt / adapter injection.

Closes the **Soft-prompt / adapter injection** spec item documented as a
gap in ``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``. This is a real
mathematically-correct LoRA implementation (Hu et al. 2021,
arxiv.org/abs/2106.09685) — a tiny baseline that future work can extend
to attention/MLP modules within larger nets.

Forward pass:

    y = base(x) + scaling * (down(x) @ up.T)
                     = base(x) + (alpha / rank) * B(A(x))

where ``A`` and ``B`` are low-rank matrices that adapt the base linear
layer without touching its weights. Base layer weights stay frozen by
default; only ``A`` and ``B`` train.

Design:
- Real torch module (requires torch at runtime); tests skip if torch
  is unavailable.
- Base layer is configurable: any ``nn.Module`` taking ``[..., in_features]``
  and returning ``[..., out_features]`` works.
- Initialisation: ``A`` is Kaiming-uniform (so the down-projection
  starts with reasonable activations), ``B`` is zero (so the adapter
  starts as a no-op and training begins from the base model's
  behaviour). Standard LoRA init.
- ``freeze_base()`` zeroes the gradient of the base layer's parameters
  for training-loop convenience.
- ``rank=0`` produces an identity adapter (forward == base) — useful
  for ablations and unit tests.
"""

from __future__ import annotations

from typing import Optional


def _torch():
    import torch  # imported lazily so the module loads without torch
    return torch


class LoRAAdapter:
    """Low-rank additive adapter for a frozen base linear layer.

    Args:
        base: a torch.nn.Module mapping (..., in_features) -> (..., out_features).
            Must expose ``in_features`` and ``out_features`` attributes
            (as ``nn.Linear`` does). If absent, pass them explicitly via
            ``in_features`` / ``out_features``.
        rank: low-rank dimension. ``0`` produces an identity adapter
            (forward equals base forward).
        alpha: LoRA scaling. The effective update magnitude is
            ``alpha / rank``. Defaults to ``rank`` so scaling = 1.0.
        in_features / out_features: override base shape detection.
    """

    def __init__(self, base, rank: int,
                 alpha: Optional[float] = None,
                 in_features: Optional[int] = None,
                 out_features: Optional[int] = None) -> None:
        torch = _torch()
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got {rank}")
        in_f = in_features if in_features is not None else getattr(base, "in_features")
        out_f = out_features if out_features is not None else getattr(base, "out_features")
        if in_f <= 0 or out_f <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(rank if rank > 0 else 1)
        self.in_features = int(in_f)
        self.out_features = int(out_f)
        self.scaling = (self.alpha / self.rank) if self.rank > 0 else 0.0
        if self.rank > 0:
            self.A = torch.nn.Parameter(torch.empty(self.rank, self.in_features))
            self.B = torch.nn.Parameter(torch.zeros(self.out_features, self.rank))
            torch.nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        else:
            self.A = None
            self.B = None

    # ------------------------------------------------------------------
    # Training-loop ergonomics
    # ------------------------------------------------------------------

    def freeze_base(self) -> None:
        """Set ``requires_grad = False`` on every base-layer parameter.

        Idempotent. After this, only ``A`` and ``B`` carry gradients —
        matching the canonical LoRA setup where the base model stays
        frozen and only adapter parameters train.
        """
        for p in self.base.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        """Iterator over the adapter's trainable parameters (A and B)."""
        if self.A is not None and self.B is not None:
            yield self.A
            yield self.B

    def num_trainable_parameters(self) -> int:
        if self.rank == 0:
            return 0
        return self.A.numel() + self.B.numel()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        torch = _torch()
        y = self.base(x)
        if self.rank == 0:
            return y
        # x: (..., in_features); flatten batch dims for matmul, then reshape.
        # A: (rank, in_features); B: (out_features, rank)
        # delta = (x @ A.T) @ B.T -> (..., out_features)
        delta = x @ self.A.t()
        delta = delta @ self.B.t()
        return y + self.scaling * delta


__all__ = ["LoRAAdapter"]
