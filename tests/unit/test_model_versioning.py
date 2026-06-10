"""Tests for ``music_brain.model_versioning``."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_brain.model_versioning import (
    CompatibilityVerdict,
    ModelManifest,
    is_compatible,
    pick_rollback_target,
    short_hash,
)


def _manifest(
    *,
    name: str = "AudioJEPA-tiny",
    schema: tuple[int, int] = (1, 0),
    arch: str = "a" * 12,
    params: str = "p" * 12,
    data: str = "d" * 12,
    created: str = "2026-01-01T00:00:00Z",
    notes: str = "",
) -> ModelManifest:
    return ModelManifest(
        name=name,
        schema_version=schema,
        arch_hash=arch,
        params_hash=params,
        data_hash=data,
        created_at=created,
        notes=notes,
    )


# ----------------------------------------------------------------------
# short_hash
# ----------------------------------------------------------------------


def test_short_hash_is_12_hex_chars() -> None:
    h = short_hash("hello")
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_short_hash_is_deterministic_and_input_sensitive() -> None:
    assert short_hash("a") == short_hash("a")
    assert short_hash("a") != short_hash("b")
    assert short_hash(b"a") == short_hash("a"), "bytes and str produce same hash"


# ----------------------------------------------------------------------
# Manifest serialisation
# ----------------------------------------------------------------------


def test_manifest_json_roundtrip() -> None:
    m = _manifest(notes="initial release")
    decoded = ModelManifest.from_json(m.to_json())
    assert decoded == m


def test_manifest_file_roundtrip(tmp_path: Path) -> None:
    m = _manifest(notes="paranoia: do not roll back past this")
    path = tmp_path / "model.manifest.json"
    m.save(path)
    loaded = ModelManifest.load(path)
    assert loaded == m


def test_manifest_rejects_malformed_schema_version() -> None:
    bad = (
        '{"name": "X", "schema_version": "1.0", "arch_hash": "x", '
        '"params_hash": "x", "data_hash": "x", "created_at": "now"}'
    )
    with pytest.raises(ValueError, match="schema_version"):
        ModelManifest.from_json(bad)


# ----------------------------------------------------------------------
# Compatibility policy
# ----------------------------------------------------------------------


def test_identical_manifests_are_compatible() -> None:
    m = _manifest()
    v = is_compatible(m, m)
    assert isinstance(v, CompatibilityVerdict)
    assert bool(v) is True


def test_name_mismatch_blocks() -> None:
    a = _manifest(name="A")
    b = _manifest(name="B")
    v = is_compatible(a, b)
    assert not v
    assert "name" in v.reason.lower()


def test_arch_hash_mismatch_blocks_even_at_same_schema() -> None:
    a = _manifest(arch="aaaaaaaaaaaa")
    b = _manifest(arch="bbbbbbbbbbbb")
    v = is_compatible(a, b)
    assert not v
    assert "arch_hash" in v.reason


def test_major_schema_bump_blocks() -> None:
    consumer = _manifest(schema=(1, 0))
    candidate = _manifest(schema=(2, 0))
    v = is_compatible(consumer, candidate)
    assert not v
    assert "major" in v.reason


def test_candidate_minor_older_than_consumer_blocks() -> None:
    consumer = _manifest(schema=(1, 3))
    candidate = _manifest(schema=(1, 2))
    v = is_compatible(consumer, candidate)
    assert not v
    assert "minor" in v.reason.lower()


def test_candidate_newer_minor_is_compatible() -> None:
    consumer = _manifest(schema=(1, 0))
    candidate = _manifest(schema=(1, 5))
    v = is_compatible(consumer, candidate)
    assert bool(v) is True


def test_params_and_data_hash_differences_are_not_blocking() -> None:
    """params_hash and data_hash differ across compatible retrainings."""
    consumer = _manifest(params="p1" * 6, data="d1" * 6)
    candidate = _manifest(params="p2" * 6, data="d2" * 6)
    assert is_compatible(consumer, candidate)


# ----------------------------------------------------------------------
# Rollback selection
# ----------------------------------------------------------------------


def test_pick_rollback_target_returns_newest_compatible() -> None:
    consumer = _manifest(schema=(1, 1))
    candidates = [
        _manifest(schema=(1, 0), created="2026-01-01T00:00:00Z"),  # too old (minor)
        _manifest(schema=(1, 1), created="2026-03-01T00:00:00Z"),  # compatible, newer
        _manifest(schema=(1, 1), created="2026-02-01T00:00:00Z"),  # compatible, older
        _manifest(schema=(2, 0), created="2026-04-01T00:00:00Z"),  # major-blocked
    ]
    picked = pick_rollback_target(consumer, candidates)
    assert picked is not None
    assert picked.created_at == "2026-03-01T00:00:00Z"


def test_pick_rollback_target_returns_none_when_nothing_compatible() -> None:
    consumer = _manifest(name="A")
    candidates = [_manifest(name="B"), _manifest(name="C")]
    assert pick_rollback_target(consumer, candidates) is None
