"""Tests for ``music_brain.streaming_wav``."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from music_brain.streaming_wav import (
    StreamingWavWriter,
    WavSpec,
    open_wav,
)

# ----------------------------------------------------------------------
# WavSpec validation
# ----------------------------------------------------------------------


def test_spec_rejects_non_positive_sample_rate() -> None:
    with pytest.raises(ValueError):
        WavSpec(sample_rate=0, num_channels=1, bits_per_sample=16, total_samples_per_channel=0)


def test_spec_rejects_non_positive_channels() -> None:
    with pytest.raises(ValueError):
        WavSpec(sample_rate=44100, num_channels=0, bits_per_sample=16, total_samples_per_channel=0)


def test_spec_only_supports_16_bit() -> None:
    with pytest.raises(ValueError, match="16-bit"):
        WavSpec(sample_rate=44100, num_channels=1, bits_per_sample=24, total_samples_per_channel=0)


# ----------------------------------------------------------------------
# Header and chunk emission
# ----------------------------------------------------------------------


def test_in_order_chunks_emit_immediately() -> None:
    spec = WavSpec(
        sample_rate=44100, num_channels=1, bits_per_sample=16, total_samples_per_channel=8
    )
    writer = StreamingWavWriter(spec)
    chunk = b"\x00\x00" * 4  # 4 frames mono
    writer.write_chunk(0, chunk)
    assert writer.next_index == 1
    writer.write_chunk(1, chunk)
    assert writer.next_index == 2
    writer.end()
    blob = writer.buffer()
    assert blob[:4] == b"RIFF"
    # Header + data
    assert len(blob) == 44 + 16


def test_out_of_order_chunks_are_buffered_then_drained() -> None:
    spec = WavSpec(
        sample_rate=44100, num_channels=1, bits_per_sample=16, total_samples_per_channel=12
    )
    writer = StreamingWavWriter(spec)
    c0 = b"\x01\x00\x02\x00"  # 2 frames
    c1 = b"\x03\x00\x04\x00"
    c2 = b"\x05\x00\x06\x00"
    # Arrive in reverse order.
    writer.write_chunk(2, c2)
    assert writer.next_index == 0
    assert writer.pending_count == 1
    writer.write_chunk(1, c1)
    assert writer.next_index == 0
    assert writer.pending_count == 2
    writer.write_chunk(0, c0)
    # All three should now have drained.
    assert writer.next_index == 3
    assert writer.pending_count == 0
    writer.end()
    blob = writer.buffer()
    data = blob[44:]
    # Bytes emitted in deterministic 0,1,2 order regardless of submit order.
    assert data == c0 + c1 + c2


def test_replaying_old_index_raises() -> None:
    spec = WavSpec(
        sample_rate=44100, num_channels=1, bits_per_sample=16, total_samples_per_channel=4
    )
    writer = StreamingWavWriter(spec)
    writer.write_chunk(0, b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="already emitted"):
        writer.write_chunk(0, b"\x00\x00\x00\x00")


def test_misaligned_chunk_size_raises() -> None:
    spec = WavSpec(
        sample_rate=44100, num_channels=2, bits_per_sample=16, total_samples_per_channel=4
    )
    writer = StreamingWavWriter(spec)
    # 5 bytes is not a whole number of 4-byte stereo frames.
    with pytest.raises(ValueError, match="not aligned"):
        writer.write_chunk(0, b"\x00\x01\x02\x03\x04")


def test_finalised_writer_refuses_more_chunks() -> None:
    spec = WavSpec(
        sample_rate=44100, num_channels=1, bits_per_sample=16, total_samples_per_channel=2
    )
    writer = StreamingWavWriter(spec)
    writer.write_chunk(0, b"\x00\x00")
    writer.end()
    with pytest.raises(RuntimeError, match="finalised"):
        writer.write_chunk(1, b"\x00\x00")


def test_begin_is_idempotent() -> None:
    spec = WavSpec(
        sample_rate=44100, num_channels=1, bits_per_sample=16, total_samples_per_channel=0
    )
    writer = StreamingWavWriter(spec)
    writer.begin()
    writer.begin()
    writer.end()
    assert writer.buffer()[:4] == b"RIFF"
    # Header written exactly once.
    assert len(writer.buffer()) == 44


def test_context_manager_lifecycle() -> None:
    spec = WavSpec(
        sample_rate=44100, num_channels=1, bits_per_sample=16, total_samples_per_channel=2
    )
    with StreamingWavWriter(spec) as w:
        w.write_chunk(0, b"\x00\x00")
        w.write_chunk(1, b"\x00\x00")
    # After exit, the writer is finalised.
    with pytest.raises(RuntimeError):
        w.write_chunk(2, b"\x00\x00")


def test_round_trip_through_stdlib_wave_module(tmp_path: Path) -> None:
    """Files written by StreamingWavWriter must be readable by stdlib wave."""
    spec = WavSpec(
        sample_rate=44100, num_channels=2, bits_per_sample=16, total_samples_per_channel=4
    )
    path = tmp_path / "test.wav"
    writer = open_wav(path, spec)
    # 4 stereo frames -> 4 * 4 = 16 bytes of PCM.
    pcm = struct.pack("<8h", 100, -100, 200, -200, 300, -300, 400, -400)
    writer.write_chunk(0, pcm)
    writer.end()
    # writer doesn't own the sink lifecycle when given a file handle,
    # so flush+close happen via gc on the BinaryIO. Force close by
    # accessing the buffer through stdlib wave below.
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 44100
        assert wf.getnchannels() == 2
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == 4
        frames = wf.readframes(4)
        assert frames == pcm


def test_deterministic_byte_output_regardless_of_submit_order() -> None:
    """Same set of chunks, two different arrival orders -> same bytes."""
    spec = WavSpec(
        sample_rate=44100, num_channels=1, bits_per_sample=16, total_samples_per_channel=6
    )
    chunks = {
        0: b"\xaa\x00\xab\x00",
        1: b"\xbb\x00\xbc\x00",
        2: b"\xcc\x00\xcd\x00",
    }

    forward = StreamingWavWriter(spec)
    for i in (0, 1, 2):
        forward.write_chunk(i, chunks[i])
    forward.end()

    reverse = StreamingWavWriter(spec)
    for i in (2, 1, 0):
        reverse.write_chunk(i, chunks[i])
    reverse.end()

    assert forward.buffer() == reverse.buffer()
