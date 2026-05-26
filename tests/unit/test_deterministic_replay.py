"""Tests for ``music_brain.deterministic_replay``."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from music_brain.deterministic_replay import (
    DeterministicContext,
    ReplayLog,
    ReplayRecord,
    _hash_payload,
)


# ----------------------------------------------------------------------
# DeterministicContext
# ----------------------------------------------------------------------


def test_same_seed_same_sequence() -> None:
    with DeterministicContext(42):
        a = [random.random() for _ in range(8)]
    with DeterministicContext(42):
        b = [random.random() for _ in range(8)]
    assert a == b


def test_different_seed_different_sequence() -> None:
    with DeterministicContext(1):
        a = [random.random() for _ in range(8)]
    with DeterministicContext(2):
        b = [random.random() for _ in range(8)]
    assert a != b


def test_exit_restores_prior_random_state() -> None:
    random.seed(12345)
    random.random()
    # Burn the next value to advance state.
    _ = random.random()
    snapshot = random.getstate()
    with DeterministicContext(999):
        # Inside the block we use a different seed.
        inside = [random.random() for _ in range(3)]
    # After exit, the prior state is restored — the next random()
    # matches what it would have been before the with-block.
    random.setstate(snapshot)
    expected_next = random.random()
    random.setstate(snapshot)  # rewind again so we can compare
    with DeterministicContext(999):
        pass
    actual_next = random.random()
    assert actual_next == expected_next
    # Sanity: the inside sequence is itself deterministic.
    with DeterministicContext(999):
        inside2 = [random.random() for _ in range(3)]
    assert inside == inside2


def test_numpy_seeded_when_available() -> None:
    np = pytest.importorskip("numpy")
    with DeterministicContext(7):
        a = np.random.rand(4)
    with DeterministicContext(7):
        b = np.random.rand(4)
    assert (a == b).all()


# ----------------------------------------------------------------------
# _hash_payload
# ----------------------------------------------------------------------


def test_hash_is_stable_across_dict_ordering() -> None:
    a = {"x": 1, "y": 2, "z": 3}
    b = {"z": 3, "y": 2, "x": 1}
    assert _hash_payload(a) == _hash_payload(b)


def test_hash_distinguishes_inputs() -> None:
    assert _hash_payload("a") != _hash_payload("b")
    assert _hash_payload({"x": 1}) != _hash_payload({"x": 2})


def test_hash_str_bytes_equivalent() -> None:
    assert _hash_payload("hello") == _hash_payload(b"hello")


# ----------------------------------------------------------------------
# ReplayRecord
# ----------------------------------------------------------------------


def test_record_json_roundtrip() -> None:
    r = ReplayRecord(
        run_id="run-1",
        seed=42,
        input_hash="abc123",
        model_hash="model-v1",
        output_hash="out-deadbeef",
        timestamp="2026-05-22T06:00:00Z",
        extra={"notes": "first pass"},
    )
    decoded = ReplayRecord.from_json(r.to_json())
    assert decoded == r


# ----------------------------------------------------------------------
# ReplayLog
# ----------------------------------------------------------------------


def _record(**overrides) -> ReplayRecord:
    defaults = dict(
        run_id="run",
        seed=1,
        input_hash="in",
        model_hash="mdl",
        output_hash="out",
        timestamp="2026-05-22T00:00:00Z",
    )
    defaults.update(overrides)
    return ReplayRecord(**defaults)


def test_append_and_iter(tmp_path: Path) -> None:
    log = ReplayLog(tmp_path / "replay.jsonl")
    log.append(_record(run_id="a"))
    log.append(_record(run_id="b"))
    records = list(log)
    assert [r.run_id for r in records] == ["a", "b"]


def test_iter_empty_when_file_missing(tmp_path: Path) -> None:
    log = ReplayLog(tmp_path / "nope.jsonl")
    assert list(log) == []


def test_find_conflicts_detects_determinism_regression(tmp_path: Path) -> None:
    log = ReplayLog(tmp_path / "replay.jsonl")
    # Historical record.
    log.append(_record(run_id="r1", input_hash="X", seed=42, model_hash="M", output_hash="OUT-A"))
    # A new record with same triplet but different output is a conflict.
    new = _record(run_id="r2", input_hash="X", seed=42, model_hash="M", output_hash="OUT-B")
    conflicts = log.find_conflicts(new)
    assert len(conflicts) == 1
    assert conflicts[0].output_hash == "OUT-A"


def test_find_conflicts_ignores_different_triplets(tmp_path: Path) -> None:
    log = ReplayLog(tmp_path / "replay.jsonl")
    log.append(_record(input_hash="X", seed=42, model_hash="M", output_hash="OUT-A"))
    # Different seed -> not a conflict, expected to differ.
    new = _record(input_hash="X", seed=99, model_hash="M", output_hash="OUT-B")
    assert log.find_conflicts(new) == []


def test_find_conflicts_ignores_identical_records(tmp_path: Path) -> None:
    log = ReplayLog(tmp_path / "replay.jsonl")
    log.append(_record(run_id="r1", output_hash="OUT-A"))
    same = _record(run_id="r2", output_hash="OUT-A")
    assert log.find_conflicts(same) == []
