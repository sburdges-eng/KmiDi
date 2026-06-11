"""Audio chunking primitives for streaming inference.

Three patterns:

  - ``iter_chunks`` — stateless generator over fixed-size windows with
    configurable hop; used to feed batched audio into models that expect
    a constant input length.
  - ``pad_to_chunk_boundary`` — make a signal a clean multiple of
    ``chunk_size``; ``mode="reflect"`` mirrors the tail to avoid the DC
    discontinuity zero-padding introduces at chunk boundaries.
  - ``overlap_add`` / :class:`WindowedOverlapAdd` — stateless and stateful
    reconstruction back to a continuous signal; useful when a model
    operates on overlapping windows and the host needs to stitch them.

All functions are stateless except ``WindowedOverlapAdd``. No ML
dependencies — works on plain NumPy arrays.

Goal-list ties: Streaming audio chunking, Chunked latent prediction
(prerequisite), Incremental decoding pipeline (prerequisite).
"""

from __future__ import annotations

from typing import Iterator, List, Literal

import numpy as np


def iter_chunks(
    signal: np.ndarray,
    chunk_size: int,
    hop_size: int,
) -> Iterator[np.ndarray]:
    """Yield successive ``chunk_size``-sample views over ``signal`` with stride ``hop_size``.

    Trailing samples that don't fill a complete chunk are dropped — use
    :func:`pad_to_chunk_boundary` first if every sample must be covered.

    Views are NumPy slices, so they alias ``signal``; copy if you need
    independent buffers downstream.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if hop_size <= 0:
        raise ValueError("hop_size must be > 0")
    signal = np.asarray(signal)
    n = signal.shape[0]
    start = 0
    while start + chunk_size <= n:
        yield signal[start : start + chunk_size]
        start += hop_size


def pad_to_chunk_boundary(
    signal: np.ndarray,
    chunk_size: int,
    mode: Literal["zero", "reflect"] = "zero",
) -> np.ndarray:
    """Pad ``signal`` along the leading axis to a multiple of ``chunk_size``.

    ``mode="zero"`` appends zeros. ``mode="reflect"`` mirrors the tail
    (avoids the DC discontinuity zero-padding can introduce).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    signal = np.asarray(signal)
    n = signal.shape[0]
    remainder = n % chunk_size
    if remainder == 0:
        return signal
    pad_len = chunk_size - remainder
    if mode == "zero":
        # Match trailing dims so (samples, channels) signals stay concatenable.
        pad = np.zeros((pad_len, *signal.shape[1:]), dtype=signal.dtype)
    elif mode == "reflect":
        # Mirror the last `pad_len` samples (excluding the last to avoid duplicates).
        # Example: [1,2,3,4,5,6] pad 2 → [5, 4]
        if pad_len > n:
            raise ValueError("reflect padding requires pad_len <= signal length")
        pad = signal[n - pad_len - 1 : n - 1][::-1].copy()
    else:
        raise ValueError(f"unsupported mode: {mode!r}")
    return np.concatenate([signal, pad])


def overlap_add(chunks: List[np.ndarray], hop_size: int) -> np.ndarray:
    """Reconstruct a continuous signal from chunks by overlap-summing.

    All chunks must have the same length. ``hop_size`` controls the stride
    between successive chunk starts; ``hop_size == chunk_size`` reverses
    non-overlapping chunking exactly.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if hop_size <= 0:
        raise ValueError("hop_size must be > 0")
    chunks = [np.asarray(c) for c in chunks]
    chunk_size = chunks[0].shape[0]
    for c in chunks:
        if c.shape[0] != chunk_size:
            raise ValueError("all chunks must have the same length")
    out_len = hop_size * (len(chunks) - 1) + chunk_size
    out = np.zeros(out_len, dtype=chunks[0].dtype)
    for i, c in enumerate(chunks):
        start = i * hop_size
        out[start : start + chunk_size] += c
    return out


class WindowedOverlapAdd:
    """Stateful streaming overlap-add reconstruction.

    Call :meth:`push` once per arriving model output chunk; it returns the
    next ``hop_size`` finalized output samples. Call :meth:`flush` at end
    of stream to drain whatever overlap is still in the buffer.

    The buffer always holds ``chunk_size`` samples; on each push the input
    chunk is summed into the buffer, the leading ``hop_size`` samples are
    emitted, and the rest is left-shifted with ``hop_size`` zeros appended
    at the tail to receive future overlaps.
    """

    def __init__(self, chunk_size: int, hop_size: int) -> None:
        if chunk_size <= 0 or hop_size <= 0:
            raise ValueError("chunk_size and hop_size must be > 0")
        if hop_size > chunk_size:
            raise ValueError("hop_size must be <= chunk_size")
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self._buf = np.zeros(chunk_size, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Add ``chunk`` into the buffer and return the next emit-window."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape[0] != self.chunk_size:
            raise ValueError(
                f"push() expected chunk of length {self.chunk_size}, got {chunk.shape[0]}"
            )
        self._buf += chunk
        out = self._buf[: self.hop_size].copy()
        # Left-shift remaining samples; pad tail with zeros for future overlap.
        new_buf = np.zeros(self.chunk_size, dtype=np.float32)
        new_buf[: self.chunk_size - self.hop_size] = self._buf[self.hop_size :]
        self._buf = new_buf
        return out

    def flush(self) -> np.ndarray:
        """Return remaining ``chunk_size - hop_size`` samples of overlap tail."""
        return self._buf[: self.chunk_size - self.hop_size].copy()
