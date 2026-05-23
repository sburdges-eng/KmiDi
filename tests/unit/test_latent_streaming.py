"""Tests for ``music_brain.latent.streaming.stream_decode``.

Closes a thin slice of the **Streaming inference support** /
**Chunked latent prediction** spec items. The full KV-cache /
incremental decoder is a follow-up PR; this hook just guarantees
deterministic in-order consumption of LatentFrame producers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.intent_ir import IntentProvenance, IntentSource  # noqa: E402
from music_brain.latent.latent_frame import LatentFrame  # noqa: E402
from music_brain.latent.streaming import stream_decode  # noqa: E402


_PROV = IntentProvenance(source=IntentSource.ML_AUDIO, user_override_weight=0.0)


def _frame(t: int, value: float = 0.0) -> LatentFrame:
    return LatentFrame(
        audio_z=torch.full((1, 4), value, dtype=torch.float32),
        chord_z=None,
        emotion_va=(0.0, 0.0),
        time_index=t,
        provenance=_PROV,
        metadata={"t": t},
    )


def test_in_order_passthrough() -> None:
    frames = [_frame(t) for t in range(4)]
    out = list(stream_decode(frames))
    assert [f.time_index for f in out] == [0, 1, 2, 3]


def test_out_of_order_buffered_then_emitted_in_order() -> None:
    src = [_frame(2), _frame(0), _frame(1), _frame(3)]
    out = list(stream_decode(src))
    assert [f.time_index for f in out] == [0, 1, 2, 3]


def test_gap_raises_after_iteration_completes() -> None:
    # Producer skipped time_index=2; consumer must surface that.
    src = [_frame(0), _frame(1), _frame(3)]
    with pytest.raises(ValueError, match="gap"):
        list(stream_decode(src))


def test_duplicate_time_index_raises() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        list(stream_decode([_frame(0), _frame(1), _frame(1)]))


def test_explicit_start_index() -> None:
    # Stream that legitimately begins mid-session at t=10.
    src = [_frame(11), _frame(10), _frame(12)]
    out = list(stream_decode(src, start_index=10))
    assert [f.time_index for f in out] == [10, 11, 12]


def test_default_start_index_picks_up_lowest_observed() -> None:
    out = list(stream_decode([_frame(5), _frame(6)]))
    assert [f.time_index for f in out] == [5, 6]


def test_on_chunk_callback_fires_in_order() -> None:
    seen: list[int] = []
    src = [_frame(2), _frame(0), _frame(1)]
    out = list(stream_decode(src, on_chunk=lambda f: seen.append(f.time_index)))
    assert seen == [0, 1, 2]
    assert [f.time_index for f in out] == [0, 1, 2]


def test_rejects_non_latent_frame() -> None:
    with pytest.raises(TypeError, match="LatentFrame"):
        list(stream_decode([_frame(0), "not a frame"]))  # type: ignore[list-item]


def test_empty_iterable_yields_nothing() -> None:
    assert list(stream_decode([])) == []


def test_yields_a_generator() -> None:
    """stream_decode is a generator so consumers can interleave with I/O."""
    import types
    gen = stream_decode([_frame(0)])
    assert isinstance(gen, types.GeneratorType)
