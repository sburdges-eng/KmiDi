"""Thin streaming hook over ``LatentFrame`` producers.

A deterministic ordered drain of an arbitrary-order LatentFrame source.
Mirrors the heap-buffered design of
``music_brain/streaming_wav.py:StreamingWavWriter._pending`` so the
audio-side and latent-side stream protocols are reasoned about the
same way: chunks are submitted in any order, the consumer emits them
in strictly monotonic ``time_index`` order, and gaps surface as errors
rather than silently dropping work.

This is intentionally *not* a full incremental decoder. The KV-cache /
constraint-sampling / sub-16ms decode loop is the next PR in the plan
(PR 2). Today's hook gives downstream code a stable contract to build
against while the heavyweight decoder is being designed.
"""

from __future__ import annotations

import heapq
from typing import Callable, Iterable, Iterator, Optional

from music_brain.latent.latent_frame import LatentFrame


def stream_decode(
    frames: Iterable[LatentFrame],
    *,
    start_index: Optional[int] = None,
    on_chunk: Optional[Callable[[LatentFrame], None]] = None,
) -> Iterator[LatentFrame]:
    """Yield ``LatentFrame``s in strictly monotonic ``time_index`` order.

    The producer may submit frames out of order; this generator buffers
    them in a min-heap and emits each as soon as it bridges the gap to
    the current expected index.

    Args:
        frames: any iterable / generator of ``LatentFrame``.
        start_index: explicit expected first ``time_index``. When ``None``,
            the first observed frame's index is used. Pass an explicit
            value when the stream legitimately begins mid-session.
        on_chunk: optional callback invoked with each frame as it is
            emitted (after ordering, before yield).

    Raises:
        TypeError: if a non-``LatentFrame`` enters the stream.
        ValueError: on a duplicate ``time_index`` or an unresolved gap
            (the producer ended without ever submitting the index the
            consumer was waiting for).
    """
    pending: list[tuple[int, int, LatentFrame]] = []  # (idx, seq, frame)
    seq = 0  # tie-break so heapq never tries to compare LatentFrames
    seen: set[int] = set()
    # When the caller pins start_index we drain incrementally as gaps
    # fill; when they don't, the producer might still submit frames
    # below the first one we observe, so we collect everything first
    # and drain at the end. Both paths share the same gap check.
    expected: Optional[int] = start_index
    streaming = start_index is not None

    def _emit(frame: LatentFrame) -> LatentFrame:
        if on_chunk is not None:
            on_chunk(frame)
        return frame

    for frame in frames:
        if not isinstance(frame, LatentFrame):
            raise TypeError(
                f"stream_decode expects LatentFrame, got {type(frame).__name__}")
        if frame.time_index in seen:
            raise ValueError(
                f"duplicate time_index {frame.time_index} in stream")
        seen.add(frame.time_index)
        if streaming and frame.time_index < expected:  # type: ignore[operator]
            raise ValueError(
                f"frame time_index {frame.time_index} is below "
                f"start_index {expected}")
        heapq.heappush(pending, (frame.time_index, seq, frame))
        seq += 1
        if streaming:
            while pending and pending[0][0] == expected:
                _, _, ready = heapq.heappop(pending)
                yield _emit(ready)
                expected += 1  # type: ignore[operator]

    # Final drain for the non-streaming path: now that we've seen every
    # frame the producer will ever submit, the lowest observed index is
    # authoritative.
    if not streaming and pending:
        expected = pending[0][0]
        while pending and pending[0][0] == expected:
            _, _, ready = heapq.heappop(pending)
            yield _emit(ready)
            expected += 1

    if pending:
        raise ValueError(
            f"stream_decode: gap at time_index {expected}; "
            f"{len(pending)} frame(s) buffered above the gap "
            f"(first stuck at index {pending[0][0]})")


__all__ = ["stream_decode"]
