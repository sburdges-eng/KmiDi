"""Cross-attention conditioning bridge.

Closes the **Cross-attention conditioning bridge** spec item documented
as a gap in ``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``.

Standard scaled dot-product cross-attention (Vaswani et al. 2017) that
treats the *intent* (emotion vector, structural plan, retrieved
memories) as the key/value source and the *audio/MIDI latents* as the
query source. Composing this layer into a generation transformer is the
canonical way to inject intent conditioning into the latent state.

This is the conditioning *bridge* — a single layer with a documented
contract. Larger architectures (multi-head attention stacks, layer
norms, feed-forward sublayers) are composed from this primitive in
downstream code.

Design:
- Real torch module; tests skip cleanly without torch.
- Multi-head: ``num_heads`` evenly divides the model dimension.
- Optional key padding mask: shape (B, S_kv), True = masked-out.
- Optional attention mask: shape (S_q, S_kv) or broadcastable;
  additive, so use ``-float('inf')`` for blocking entries.
- Returns ``(output, attention_weights)`` where attention_weights is
  averaged over heads (mean of softmax outputs) for downstream
  diagnostics / debugging.
"""

from __future__ import annotations

import math
from typing import Optional


def _torch():
    import torch
    return torch


class CrossAttention:
    """Scaled dot-product cross-attention over (query, kv) pairs.

    Args:
        query_dim: dimension of the query tensor's last axis.
        kv_dim: dimension of the key/value source last axis.
            ``None`` means equal to ``query_dim``.
        num_heads: number of attention heads. Must divide ``query_dim``.
        dropout: attention-dropout probability (training-time only;
            this module honours ``module.training`` if set on the
            underlying nn.Module subclass; here we expose it but apply
            it via ``torch.nn.functional.dropout`` at call sites that
            choose to set ``training=True``).
    """

    def __init__(self, query_dim: int, *,
                 kv_dim: Optional[int] = None,
                 num_heads: int = 4,
                 dropout: float = 0.0) -> None:
        torch = _torch()
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        if num_heads <= 0 or query_dim % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must evenly divide query_dim "
                f"({query_dim})")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        kvd = query_dim if kv_dim is None else int(kv_dim)
        self.query_dim = int(query_dim)
        self.kv_dim = kvd
        self.num_heads = int(num_heads)
        self.head_dim = self.query_dim // self.num_heads
        self.dropout = float(dropout)

        # Three projection matrices. Wrapped in nn.Linear for parameter
        # tracking and standard init.
        self.q_proj = torch.nn.Linear(self.query_dim, self.query_dim, bias=False)
        self.k_proj = torch.nn.Linear(self.kv_dim, self.query_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.kv_dim, self.query_dim, bias=False)
        self.out_proj = torch.nn.Linear(self.query_dim, self.query_dim, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, query, kv,
                 key_padding_mask=None,
                 attn_mask=None,
                 training: bool = False):
        return self.forward(query, kv,
                            key_padding_mask=key_padding_mask,
                            attn_mask=attn_mask,
                            training=training)

    def forward(self, query, kv,
                key_padding_mask=None,
                attn_mask=None,
                training: bool = False):
        """Compute cross-attention.

        Shapes:
            query: (B, S_q, query_dim)
            kv:    (B, S_kv, kv_dim)
            key_padding_mask: (B, S_kv), True = mask out
            attn_mask: (S_q, S_kv), additive

        Returns:
            (output, attn_weights):
                output: (B, S_q, query_dim)
                attn_weights: (B, S_q, S_kv), averaged over heads
        """
        torch = _torch()
        b, s_q, _ = query.shape
        _, s_kv, _ = kv.shape

        q = self.q_proj(query).view(b, s_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(b, s_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(b, s_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: q (B, H, S_q, d); k (B, H, S_kv, d); v (B, H, S_kv, d)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S_q, S_kv)

        if attn_mask is not None:
            # attn_mask broadcasts over (B, H).
            scores = scores + attn_mask

        if key_padding_mask is not None:
            # (B, 1, 1, S_kv) -> True positions become -inf
            pad = key_padding_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(pad, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        if training and self.dropout > 0.0:
            attn = torch.nn.functional.dropout(attn, p=self.dropout)

        ctx = torch.matmul(attn, v)  # (B, H, S_q, d)
        ctx = ctx.transpose(1, 2).contiguous().view(b, s_q, self.query_dim)
        out = self.out_proj(ctx)

        # Average attention across heads for diagnostics.
        attn_weights = attn.mean(dim=1)  # (B, S_q, S_kv)
        return out, attn_weights

    # ------------------------------------------------------------------
    # Parameter convenience
    # ------------------------------------------------------------------

    def parameters(self):
        yield from self.q_proj.parameters()
        yield from self.k_proj.parameters()
        yield from self.v_proj.parameters()
        yield from self.out_proj.parameters()


__all__ = ["CrossAttention"]
