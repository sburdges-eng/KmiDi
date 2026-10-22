"""Tests for ``music_brain.vector_store.VectorStore``."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from music_brain.vector_store import VectorStore


def test_dim_must_be_positive() -> None:
    with pytest.raises(ValueError):
        VectorStore(0)
    with pytest.raises(ValueError):
        VectorStore(-1)


def test_add_and_get_roundtrip() -> None:
    store = VectorStore(dim=4)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    store.add("a", v, metadata={"kind": "test"})
    got = store.get("a")
    assert got is not None
    vec_out, meta_out = got
    assert vec_out.shape == (4,)
    # Stored vectors are L2-normalised; the input was already unit-length.
    np.testing.assert_allclose(vec_out, v, atol=1e-6)
    assert meta_out == {"kind": "test"}
    assert len(store) == 1
    assert "a" in store


def test_dim_mismatch_raises() -> None:
    store = VectorStore(dim=4)
    with pytest.raises(ValueError):
        store.add("a", np.zeros(3, dtype=np.float32))


def test_add_replaces_existing_id_in_place() -> None:
    store = VectorStore(dim=2)
    store.add("a", np.array([1.0, 0.0]))
    store.add("b", np.array([0.0, 1.0]))
    # Replace 'a' — length must stay 2, 'b' must still be present.
    store.add("a", np.array([0.0, 1.0]), metadata={"v": 2})
    assert len(store) == 2
    got = store.get("a")
    assert got is not None
    _, meta = got
    assert meta == {"v": 2}


def test_search_returns_top_k_ranked_by_cosine() -> None:
    store = VectorStore(dim=3)
    # Three orthogonal axes.
    store.add("x", np.array([1.0, 0.0, 0.0]))
    store.add("y", np.array([0.0, 1.0, 0.0]))
    store.add("z", np.array([0.0, 0.0, 1.0]))
    # Query close to x-axis.
    results = store.search(np.array([0.9, 0.1, 0.05]), top_k=2)
    assert len(results) == 2
    ids = [r[0] for r in results]
    assert ids[0] == "x", "Closest neighbour should rank first"
    # Scores in [-1, 1], descending.
    assert results[0][1] >= results[1][1]


def test_search_respects_min_score_threshold() -> None:
    store = VectorStore(dim=2)
    store.add("near", np.array([1.0, 0.0]))
    store.add("orthogonal", np.array([0.0, 1.0]))
    store.add("opposite", np.array([-1.0, 0.0]))
    # Query along positive x.
    out = store.search(np.array([1.0, 0.0]), top_k=10, min_score=0.5)
    ids = [r[0] for r in out]
    assert "near" in ids
    assert "orthogonal" not in ids  # score = 0.0
    assert "opposite" not in ids  # score = -1.0


def test_search_on_empty_store_returns_empty() -> None:
    store = VectorStore(dim=4)
    assert store.search(np.zeros(4)) == []


def test_delete_removes_entry_and_shifts_rows() -> None:
    store = VectorStore(dim=2)
    store.add("a", np.array([1.0, 0.0]))
    store.add("b", np.array([0.0, 1.0]))
    store.add("c", np.array([1.0, 1.0]))
    assert store.delete("a") is True
    assert len(store) == 2
    assert store.get("a") is None
    # 'b' and 'c' are still searchable.
    out = store.search(np.array([0.0, 1.0]), top_k=2)
    ids = [r[0] for r in out]
    assert "b" in ids
    assert "c" in ids
    # Re-deleting is a no-op.
    assert store.delete("a") is False


def test_persistence_roundtrip(tmp_path: Path) -> None:
    store = VectorStore(dim=3)
    store.add("alpha", np.array([1.0, 0.0, 0.0]), metadata={"tag": "A"})
    store.add("beta", np.array([0.0, 1.0, 0.0]), metadata={"tag": "B"})
    path = tmp_path / "store.npz"
    store.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == 2
    assert "alpha" in loaded
    assert "beta" in loaded
    got = loaded.get("beta")
    assert got is not None
    vec, meta = got
    np.testing.assert_allclose(vec, np.array([0.0, 1.0, 0.0]), atol=1e-6)
    assert meta == {"tag": "B"}
    # Search still works after load.
    out = loaded.search(np.array([1.0, 0.0, 0.0]), top_k=1)
    assert out[0][0] == "alpha"


def test_persistence_preserves_empty_store(tmp_path: Path) -> None:
    store = VectorStore(dim=8)
    path = tmp_path / "empty.npz"
    store.save(path)
    loaded = VectorStore.load(path)
    assert loaded.dim == 8
    assert len(loaded) == 0
    assert loaded.search(np.zeros(8)) == []


@pytest.mark.parametrize("empty", [False, True])
def test_saved_archive_is_readable_without_pickle(tmp_path: Path, empty: bool) -> None:
    store = VectorStore(dim=2)
    if not empty:
        store.add("entry", np.array([3.0, 4.0]), metadata={"tag": "value"})
    path = tmp_path / "safe.npz"
    store.save(path)
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            assert not data[name].dtype.hasobject
        assert data["ids"].dtype.kind == "U"
        assert data["meta"].dtype.kind == "U"
        assert data["vectors"].shape == (0 if empty else 1, 2)
        assert data["vectors"].dtype == np.float32


def test_persistence_preserves_ids_metadata_and_rebuilds_index(tmp_path: Path) -> None:
    store = VectorStore(dim=2)
    store.add("removed", np.array([1.0, 0.0]))
    ids = ["a", "音楽-🎵" * 100, 'quoted"\\id\x00', "no-metadata"]
    metadata = [
        {},
        {"nested": {"tags": ["café", None, True, 0, 0.25]}, "text": "line\n\x00"},
        {"empty": "", "list": []},
        None,
    ]
    for id_, meta in zip(ids, metadata):
        store.add(id_, np.array([3.0, 4.0]), metadata=meta)
    assert store.delete("removed")
    path = tmp_path / "nested" / "store.npz"
    store.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == 4
    assert "removed" not in loaded
    for id_, meta in zip(ids, metadata):
        assert id_ in loaded
        vec, restored_meta = loaded.get(id_)
        np.testing.assert_allclose(vec, [0.6, 0.8], atol=1e-6)
        assert restored_meta == meta
    assert {hit[0] for hit in loaded.search(np.array([3.0, 4.0]))} == set(ids)
    loaded.add(ids[1], np.array([0.0, 1.0]), metadata={"updated": True})
    assert len(loaded) == 4
    assert loaded.get(ids[1])[1] == {"updated": True}
    assert loaded.delete(ids[2])
    assert loaded.get(ids[3])[1] is None


class _PickleProbe:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        # Harmless marker: this must never execute during archive loading.
        return Path.touch, (self.marker,)


@pytest.mark.parametrize("field", ["dim", "vectors", "ids", "meta"])
def test_load_rejects_object_arrays_without_executing_pickle(tmp_path: Path, field: str) -> None:
    marker = tmp_path / "unpickled"
    arrays = {
        "dim": np.asarray(2),
        "vectors": np.array([[1.0, 0.0]], dtype=np.float32),
        "ids": np.asarray([json.dumps("a")]),
        "meta": np.asarray([""]),
    }
    arrays[field] = np.asarray([_PickleProbe(marker)], dtype=object)
    path = tmp_path / "untrusted.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="[Oo]bject.*pickle"):
        VectorStore.load(path)
    assert not marker.exists()


def test_load_rejects_raw_pickle_without_executing_it(tmp_path: Path) -> None:
    marker = tmp_path / "unpickled"
    path = tmp_path / "disguised.npz"
    path.write_bytes(pickle.dumps(_PickleProbe(marker)))
    with pytest.raises(ValueError, match="pickle"):
        VectorStore.load(path)
    assert not marker.exists()


def test_load_rejects_missing_arrays(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, dim=np.asarray(2))
    with pytest.raises(ValueError, match="missing a required array"):
        VectorStore.load(path)


@pytest.mark.parametrize("empty", [False, True])
def test_load_explicitly_rejects_legacy_archives(tmp_path: Path, empty: bool) -> None:
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        dim=np.asarray(2),
        vectors=np.zeros((0 if empty else 1, 2), dtype=np.float32),
        ids=np.asarray([] if empty else ["a"], dtype=object),
        meta=np.asarray([] if empty else ['{"tag": "old"}'], dtype=object),
    )
    with pytest.raises(ValueError, match="[Oo]bject.*pickle"):
        VectorStore.load(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dim", np.asarray([2])),
        ("dim", np.asarray(2.5)),
        ("dim", np.asarray(0)),
        ("vectors", np.zeros(2)),
        ("vectors", np.zeros((2, 3))),
        ("vectors", np.zeros((0, 3))),
        ("ids", np.asarray([json.dumps("a")])),
        ("ids", np.asarray([[json.dumps("a"), json.dumps("b")]])),
        ("ids", np.asarray([1, 2])),
        ("ids", np.asarray(['"a"', '"a"'])),
        ("ids", np.asarray(['""', '"b"'])),
        ("ids", np.asarray(["1", '"b"'])),
        ("ids", np.asarray(["invalid-json", '"b"'])),
        ("meta", np.asarray([""])),
        ("meta", np.asarray([["", ""]])),
        ("meta", np.asarray([1, 2])),
        ("meta", np.asarray(["not-json", ""])),
        ("meta", np.asarray(["[]", ""])),
    ],
)
def test_load_rejects_malformed_archive(tmp_path: Path, field: str, value: np.ndarray) -> None:
    arrays = {
        "dim": np.asarray(2),
        "vectors": np.eye(2, dtype=np.float32),
        "ids": np.asarray([json.dumps("a"), json.dumps("b")]),
        "meta": np.asarray(["", "{}"]),
    }
    arrays[field] = value
    path = tmp_path / "malformed.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        VectorStore.load(path)


def test_zero_vector_does_not_break_normalisation() -> None:
    store = VectorStore(dim=3)
    # Zero vector is degenerate; we store it as-is (no NaN) and rely on
    # search semantics returning low scores against it.
    store.add("zero", np.zeros(3, dtype=np.float32))
    got = store.get("zero")
    assert got is not None
    vec, _ = got
    assert not np.any(np.isnan(vec))
