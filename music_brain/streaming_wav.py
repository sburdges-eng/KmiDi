"""Streaming chunked WAV writer with deterministic ordering.

Closes the **Streaming WAV export** / **Fully-streaming WAV export**
spec item documented as a gap in
``docs/audit/RT_CALLBACK_AUDIT_2026-05-22.md``. Reinforces
``docs/ARCHITECTURAL_CONTRACTS.md`` §12 (Clear export determinism).

Why this exists:

The stdlib ``wave`` module requires the full audio buffer to be assembled
in memory before writing the RIFF header (because the header needs the
total byte count). For long-form generation that streams chunks from an
inference worker, that breaks the determinism contract: either we buffer
everything (defeats streaming), or we patch the header after-the-fact
(introduces nondeterministic write ordering and breaks naive readers
mid-stream).

This module writes WAVE files in two phases:

  1. ``begin()`` writes a RIFF header with a *known total size*
     supplied by the caller. The caller computes this from
     ``samples_per_chunk * num_chunks`` ahead of time. That keeps the
     write order strictly forward — no seek-back-and-fix, no
     post-process pass.

  2. ``write_chunk(samples)`` appends one chunk of PCM data. Chunks
     must arrive in monotonically-increasing order keyed by
     ``chunk_index`` so concurrent inference workers stream into a
     single bounded queue and the writer emits in deterministic order.

Stdlib only. Format: 16-bit signed PCM, little-endian, mono or
multi-channel interleaved.
"""

from __future__ import annotations

import heapq
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

_RIFF_HEADER_SIZE = 44


@dataclass(frozen=True)
class WavSpec:
    """Audio format spec for the streaming writer.

    Args:
        sample_rate: e.g. 44100, 48000.
        num_channels: 1 for mono, 2 for stereo, etc.
        bits_per_sample: only 16 supported in this minimal writer.
        total_samples_per_channel: planned length, used to fill the RIFF
            header up-front. If the actual stream is shorter, the file
            remains a valid WAVE (extra header bytes describe space that
            is not written), but most readers will report EOF early.
    """

    sample_rate: int
    num_channels: int
    bits_per_sample: int
    total_samples_per_channel: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.bits_per_sample != 16:
            raise ValueError("only 16-bit PCM is supported")
        if self.total_samples_per_channel < 0:
            raise ValueError("total_samples_per_channel must be >= 0")

    @property
    def bytes_per_frame(self) -> int:
        return self.num_channels * (self.bits_per_sample // 8)

    @property
    def total_data_bytes(self) -> int:
        return self.total_samples_per_channel * self.bytes_per_frame


def _pack_riff_header(spec: WavSpec) -> bytes:
    """Build the 44-byte RIFF header for ``spec``."""
    byte_rate = spec.sample_rate * spec.bytes_per_frame
    data_size = spec.total_data_bytes
    riff_size = 36 + data_size
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)  # subchunk1 size
        + struct.pack("<H", 1)  # audio format = PCM
        + struct.pack("<H", spec.num_channels)
        + struct.pack("<I", spec.sample_rate)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", spec.bytes_per_frame)
        + struct.pack("<H", spec.bits_per_sample)
        + b"data"
        + struct.pack("<I", data_size)
    )


def _check_pcm_chunk(spec: WavSpec, samples: bytes) -> None:
    if len(samples) % spec.bytes_per_frame != 0:
        raise ValueError(
            f"chunk size {len(samples)} not aligned to frame size " f"{spec.bytes_per_frame}"
        )


class StreamingWavWriter:
    """Append PCM chunks in a strictly forward, deterministic order.

    Out-of-order chunks (by ``chunk_index``) are buffered until the gap
    is filled, then drained in order. This lets multiple inference
    workers run in parallel and submit results back in any order — the
    writer still emits a byte-stable file.
    """

    def __init__(
        self, spec: WavSpec, sink: Optional[BinaryIO] = None, *, owns_sink: bool = False
    ) -> None:
        self._spec = spec
        self._sink = sink or io.BytesIO()
        self._owns_sink = owns_sink
        self._started = False
        self._finalised = False
        self._next_index = 0
        self._pending: list[tuple[int, bytes]] = []  # heap by chunk_index
        self._bytes_written = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin(self) -> None:
        """Write the RIFF header. Idempotent: second call is a no-op."""
        if self._started:
            return
        self._sink.write(_pack_riff_header(self._spec))
        self._started = True

    def write_chunk(self, chunk_index: int, samples: bytes) -> None:
        """Submit one chunk of interleaved PCM at ``chunk_index``.

        Chunks may arrive out of order; they are buffered and drained
        contiguously as the index gap closes. Calling write_chunk before
        begin() implicitly calls begin().
        """
        if self._finalised:
            raise RuntimeError("writer is finalised; cannot accept more chunks")
        if chunk_index < self._next_index:
            raise ValueError(
                f"chunk_index {chunk_index} already emitted "
                f"(next expected = {self._next_index})"
            )
        _check_pcm_chunk(self._spec, samples)
        if not self._started:
            self.begin()
        if chunk_index == self._next_index:
            self._emit(samples)
            self._drain_buffered()
        else:
            heapq.heappush(self._pending, (chunk_index, samples))

    def end(self) -> None:
        """Mark the stream finished. No more chunks may be submitted.

        Pending out-of-order chunks that were never bridged remain
        unwritten — caller can detect this via ``pending_count``.

        Flushes the sink, and closes it when this writer owns its
        lifecycle (created via ``open_wav``).
        """
        self._finalised = True
        # Flush so a reader opening the path immediately after end()
        # sees a complete file, even if we don't own the handle.
        flush = getattr(self._sink, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:  # noqa: BLE001 — best-effort
                pass
        if self._owns_sink:
            close = getattr(self._sink, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 — best-effort
                    pass

    def __enter__(self) -> "StreamingWavWriter":
        self.begin()
        return self

    def __exit__(self, *_exc) -> None:
        self.end()

    # ------------------------------------------------------------------
    # Introspection (tests / observability)
    # ------------------------------------------------------------------

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    @property
    def next_index(self) -> int:
        return self._next_index

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def buffer(self) -> bytes:
        """Return accumulated bytes when sink is an in-memory BytesIO."""
        if not isinstance(self._sink, io.BytesIO):
            raise TypeError("buffer() requires a BytesIO sink")
        return self._sink.getvalue()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, samples: bytes) -> None:
        self._sink.write(samples)
        self._bytes_written += len(samples)
        self._next_index += 1

    def _drain_buffered(self) -> None:
        while self._pending and self._pending[0][0] == self._next_index:
            _, samples = heapq.heappop(self._pending)
            self._emit(samples)


def open_wav(path: Path, spec: WavSpec) -> StreamingWavWriter:
    """Open ``path`` for streaming write. Caller must call ``end()``.

    The returned writer owns the file handle: ``end()`` (and exiting
    the writer as a context manager) flushes and closes it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sink = path.open("wb")
    return StreamingWavWriter(spec, sink=sink, owns_sink=True)


__all__ = ["StreamingWavWriter", "WavSpec", "open_wav"]
