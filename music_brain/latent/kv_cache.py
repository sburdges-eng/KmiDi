"""Append/truncate key-value cache for incremental decoding.

Closes the **KV-cache optimization** spec item. The cache owns a
pre-allocated ``(1, num_heads, max_len, head_dim)`` buffer pair (one
for K, one for V) and tracks a current write head. ``append`` copies
incoming ``(B, H, T_new, D)`` tensors into the next slice and returns
a *view* of the populated prefix ``(B, H, length, D)`` — the
downstream cross/self-attention can consume that directly without
materializing a full reallocated tensor each step.

Truncation is free (just rewinds the write head) so the realtime
rollback ring (see ``rollback.py``) can revert to a checkpoint in O(1).

Why pre-allocate: incremental decoders that grow buffers via
``torch.cat`` allocate and free on the hot path, blowing the sub-16ms
latency target. Pre-allocating once and writing into slices keeps the
inference loop allocation-free after the first step.
"""

from __future__ import annotations

from typing import Tuple


def _torch():
    import torch
    return torch


class KVCache:
    """Pre-allocated incremental KV-cache.

    Args:
        num_heads: number of attention heads.
        head_dim: per-head feature dim.
        max_len: maximum context length the cache can hold.
        batch_size: leading batch dim. Defaults to 1 — typical for
            single-stream realtime decoding.
    """

    def __init__(self, num_heads: int, head_dim: int, max_len: int,
                 batch_size: int = 1) -> None:
        torch = _torch()
        if num_heads <= 0 or head_dim <= 0 or max_len <= 0 or batch_size <= 0:
            raise ValueError("num_heads, head_dim, max_len, batch_size must be positive")
        self._num_heads = int(num_heads)
        self._head_dim = int(head_dim)
        self._max_len = int(max_len)
        self._batch_size = int(batch_size)
        self._k = torch.zeros(self._batch_size, self._num_heads,
                              self._max_len, self._head_dim, dtype=torch.float32)
        self._v = torch.zeros_like(self._k)
        self._length = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        return self._length

    @property
    def capacity(self) -> int:
        return self._max_len

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, k_new, v_new) -> Tuple:
        """Append ``(B, H, T_new, D)`` slices and return the populated prefix.

        Returns:
            (k_prefix, v_prefix): views of shape ``(B, H, length, D)``
            into the cache buffer. They share storage with the cache —
            ``snapshot`` instead if you need an owned copy.
        """
        if k_new.shape != v_new.shape:
            raise ValueError(
                f"k/v shape mismatch: {tuple(k_new.shape)} vs {tuple(v_new.shape)}")
        if k_new.shape[1] != self._num_heads:
            raise ValueError(
                f"num_heads mismatch: cache={self._num_heads}, got {k_new.shape[1]}")
        if k_new.shape[3] != self._head_dim:
            raise ValueError(
                f"head_dim mismatch: cache={self._head_dim}, got {k_new.shape[3]}")
        t_new = int(k_new.shape[2])
        if self._length + t_new > self._max_len:
            raise ValueError(
                f"capacity exceeded: length={self._length} + new={t_new} > "
                f"max_len={self._max_len}")
        self._k[:, :, self._length:self._length + t_new, :] = k_new
        self._v[:, :, self._length:self._length + t_new, :] = v_new
        self._length += t_new
        return (self._k[:, :, :self._length, :],
                self._v[:, :, :self._length, :])

    def truncate(self, new_length: int) -> None:
        """Rewind the write head to ``new_length`` (O(1))."""
        if new_length < 0 or new_length > self._length:
            raise ValueError(
                f"truncate: target {new_length} not in [0, current={self._length}]")
        self._length = int(new_length)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def snapshot(self):
        """Return an owned copy of the populated prefix."""
        k = self._k[:, :, :self._length, :].clone()
        v = self._v[:, :, :self._length, :].clone()
        return k, v


__all__ = ["KVCache"]
