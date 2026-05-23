"""Bounded checkpoint ring for realtime rollback / audition-safe regen.

Closes **Realtime rollback system** and **Realtime audition-safe
regeneration**. A composer / human-in-the-loop UI can:

  1. ``checkpoint(frame, kv_cache=…)`` *before* trying a speculative
     branch (a new motif, a tension excursion, …).
  2. Audition the result.
  3. Either accept it (keep going — the checkpoint remains as part of
     history) or ``rollback_to(time_index)`` to revert to that branch
     point.

Each checkpoint stores the ``LatentFrame`` *and* the KV-cache length
at the time of capture. ``rollback_to`` rewinds the cache via
``KVCache.truncate`` so the inference loop can continue without
reallocating.

This is intentionally a small, O(1) primitive — the realtime hot path
must not have a malloc on rollback.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from music_brain.latent.kv_cache import KVCache
from music_brain.latent.latent_frame import LatentFrame


class _Checkpoint:
    __slots__ = ("frame", "cache_length", "cache_ref")

    def __init__(self, frame: LatentFrame, cache_length: int, cache_ref: Optional[KVCache]) -> None:
        self.frame = frame
        self.cache_length = cache_length
        self.cache_ref = cache_ref


class RollbackRing:
    """O(1) checkpoint / restore ring.

    Args:
        capacity: maximum number of checkpoints to retain. Older
            checkpoints are evicted FIFO.
    """

    def __init__(self, capacity: int = 16) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._buf: deque[_Checkpoint] = deque(maxlen=int(capacity))
        self._capacity = int(capacity)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def head_index(self) -> Optional[int]:
        if not self._buf:
            return None
        return self._buf[-1].frame.time_index

    def snapshot(self) -> list[LatentFrame]:
        """Return a shallow copy of the current ring as LatentFrames."""
        return [c.frame for c in self._buf]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def checkpoint(self, frame: LatentFrame, *, kv_cache: Optional[KVCache] = None) -> None:
        if self._buf and frame.time_index <= self._buf[-1].frame.time_index:
            raise ValueError(
                f"checkpoint time_index must be monotonic; "
                f"got {frame.time_index} <= head {self._buf[-1].frame.time_index}"
            )
        self._buf.append(
            _Checkpoint(
                frame=frame,
                cache_length=kv_cache.length if kv_cache is not None else 0,
                cache_ref=kv_cache,
            )
        )

    def rollback_to(self, time_index: int) -> LatentFrame:
        """Discard checkpoints after ``time_index`` and return that frame.

        Also truncates any KV-cache that was attached to that
        checkpoint back to its captured length.
        """
        target_idx = None
        for i, c in enumerate(self._buf):
            if c.frame.time_index == time_index:
                target_idx = i
                break
        if target_idx is None:
            raise KeyError(f"no checkpoint at time_index {time_index}")
        target = self._buf[target_idx]
        # Drop everything strictly after the target.
        while len(self._buf) > target_idx + 1:
            self._buf.pop()
        # Restore cache state.
        if target.cache_ref is not None:
            target.cache_ref.truncate(target.cache_length)
        return target.frame


__all__ = ["RollbackRing"]
