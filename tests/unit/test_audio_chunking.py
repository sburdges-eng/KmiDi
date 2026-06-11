"""Audio chunking primitives (Goal: Streaming audio chunking,
Chunked latent prediction, Incremental decoding pipeline)."""

import numpy as np
import pytest

from music_brain.audio.chunking import (
    WindowedOverlapAdd,
    iter_chunks,
    overlap_add,
    pad_to_chunk_boundary,
)

# ---------------------------------------------------------------------------
# iter_chunks
# ---------------------------------------------------------------------------


def test_iter_chunks_non_overlapping_covers_whole_signal():
    signal = np.arange(16, dtype=np.float32)
    chunks = list(iter_chunks(signal, chunk_size=4, hop_size=4))
    assert len(chunks) == 4
    assert chunks[0].tolist() == [0, 1, 2, 3]
    assert chunks[-1].tolist() == [12, 13, 14, 15]


def test_iter_chunks_with_overlap_yields_overlapping_chunks():
    signal = np.arange(10, dtype=np.float32)
    chunks = list(iter_chunks(signal, chunk_size=4, hop_size=2))
    # Starts: 0, 2, 4, 6 → last full chunk starts at 6 (covers 6..9).
    assert [c[0] for c in chunks] == [0, 2, 4, 6]
    assert chunks[0].tolist() == [0, 1, 2, 3]
    assert chunks[1].tolist() == [2, 3, 4, 5]


def test_iter_chunks_drops_trailing_partial_when_not_padded():
    signal = np.arange(10, dtype=np.float32)
    chunks = list(iter_chunks(signal, chunk_size=4, hop_size=4))
    # 10 / 4 = 2 full chunks; samples 8-9 dropped (no partial chunks by default).
    assert len(chunks) == 2
    assert chunks[-1].tolist() == [4, 5, 6, 7]


def test_iter_chunks_invalid_params_raise():
    signal = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError):
        list(iter_chunks(signal, chunk_size=0, hop_size=4))
    with pytest.raises(ValueError):
        list(iter_chunks(signal, chunk_size=4, hop_size=0))


# ---------------------------------------------------------------------------
# pad_to_chunk_boundary
# ---------------------------------------------------------------------------


def test_pad_to_chunk_boundary_zero_extends_to_multiple_of_chunk_size():
    signal = np.arange(10, dtype=np.float32)
    padded = pad_to_chunk_boundary(signal, chunk_size=4, mode="zero")
    assert padded.shape == (12,)
    assert padded[10] == 0.0
    assert padded[11] == 0.0
    assert padded[:10].tolist() == signal.tolist()


def test_pad_to_chunk_boundary_no_op_when_already_aligned():
    signal = np.arange(8, dtype=np.float32)
    padded = pad_to_chunk_boundary(signal, chunk_size=4, mode="zero")
    assert padded is signal or padded.shape == signal.shape
    assert padded.tolist() == signal.tolist()


def test_pad_to_chunk_boundary_reflect_mode():
    """``reflect`` pads by mirroring the tail."""
    signal = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    padded = pad_to_chunk_boundary(signal, chunk_size=4, mode="reflect")
    assert padded.shape == (8,)
    # Last two samples should mirror samples 5 and 4 (reverse order):
    #   [1, 2, 3, 4, 5, 6, 5, 4]
    assert padded[6] == 5.0
    assert padded[7] == 4.0


# ---------------------------------------------------------------------------
# overlap_add (stateless)
# ---------------------------------------------------------------------------


def test_overlap_add_inverts_non_overlapping_chunks():
    signal = np.arange(12, dtype=np.float32)
    chunks = list(iter_chunks(signal, chunk_size=4, hop_size=4))
    out = overlap_add(chunks, hop_size=4)
    assert out.tolist() == signal.tolist()


def test_overlap_add_sums_overlapping_regions():
    """Two unit-valued chunks of length 4 with hop 2 → centre region has value 2."""
    chunks = [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]
    out = overlap_add(chunks, hop_size=2)
    # Layout: [1, 1, 2, 2, 1, 1]
    assert out.tolist() == [1.0, 1.0, 2.0, 2.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# WindowedOverlapAdd (stateful streaming)
# ---------------------------------------------------------------------------


def test_windowed_overlap_add_emits_consistent_hop_size_blocks():
    """Stateful streaming OLA: each push(chunk) returns hop_size samples
    of finalized output."""
    chunk_size = 4
    hop_size = 2
    ola = WindowedOverlapAdd(chunk_size=chunk_size, hop_size=hop_size)
    # Push four unit-valued chunks; each should produce ``hop_size`` samples
    # of fully overlapped output (=2 after the first one steady-states).
    outputs = []
    for _ in range(4):
        block = ola.push(np.ones(chunk_size, dtype=np.float32))
        outputs.append(block)
    # First push has nothing fully overlapped yet → returns hop_size samples
    # of the leading chunk (value 1.0).
    assert len(outputs[0]) == hop_size
    # Subsequent pushes have a second chunk overlapping → value 2.0.
    assert outputs[1].tolist() == [2.0, 2.0]
    assert outputs[2].tolist() == [2.0, 2.0]


def test_windowed_overlap_add_flush_emits_remaining_tail():
    """flush() returns whatever is still in the buffer at end of stream."""
    chunk_size = 4
    hop_size = 2
    ola = WindowedOverlapAdd(chunk_size=chunk_size, hop_size=hop_size)
    ola.push(np.ones(chunk_size, dtype=np.float32))
    ola.push(np.ones(chunk_size, dtype=np.float32))
    tail = ola.flush()
    # After 2 pushes and 2 emits of hop_size, there are chunk_size - hop_size
    # = 2 samples of overlap left in buffer.
    assert len(tail) == chunk_size - hop_size


def test_pad_to_chunk_boundary_zero_pads_multichannel():
    stereo = np.ones((5, 2), dtype=np.float32)
    padded = pad_to_chunk_boundary(stereo, chunk_size=4, mode="zero")
    assert padded.shape == (8, 2)
    assert np.all(padded[:5] == 1.0)
    assert np.all(padded[5:] == 0.0)


def test_pad_to_chunk_boundary_reflect_pads_multichannel():
    stereo = np.arange(12, dtype=np.float32).reshape(6, 2)
    padded = pad_to_chunk_boundary(stereo, chunk_size=4, mode="reflect")
    assert padded.shape == (8, 2)
    # Mirror of the two rows before the last: rows 4, 3.
    assert np.array_equal(padded[6], stereo[4])
    assert np.array_equal(padded[7], stereo[3])
