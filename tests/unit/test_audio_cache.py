"""Tests for ``music_brain.audio_cache.AudioCache``."""

from __future__ import annotations

from pathlib import Path


from music_brain.audio_cache import AudioCache


def test_key_is_deterministic(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path)
    k1 = cache.key_for(b"abc", "spectrogram-v1")
    k2 = cache.key_for(b"abc", "spectrogram-v1")
    k3 = cache.key_for(b"abc", "spectrogram-v2")
    k4 = cache.key_for(b"abcd", "spectrogram-v1")
    assert k1 == k2
    assert k1 != k3, "analyzer-id must affect the key"
    assert k1 != k4, "source bytes must affect the key"


def test_put_then_get_roundtrip(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path)
    key = cache.key_for(b"audio-bytes", "analyzer-1")
    payload = b"\x00\x01\x02 some serialized spectrogram"
    assert cache.get(key) is None
    cache.put(key, payload)
    assert cache.get(key) == payload


def test_get_missing_returns_none(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path)
    assert cache.get("does-not-exist") is None


def test_lru_eviction_drops_oldest(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path, max_entries=3)
    for i in range(4):
        cache.put(f"k{i}", f"payload-{i}".encode())
    # k0 should have been evicted; k1..k3 remain
    assert cache.get("k0") is None
    for i in range(1, 4):
        assert cache.get(f"k{i}") == f"payload-{i}".encode()


def test_get_promotes_to_mru(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path, max_entries=3)
    cache.put("a", b"1")
    cache.put("b", b"2")
    cache.put("c", b"3")
    # Touch 'a' -> now MRU. Inserting 'd' should evict 'b' (oldest after touch).
    assert cache.get("a") == b"1"
    cache.put("d", b"4")
    assert cache.get("a") == b"1"
    assert cache.get("b") is None
    assert cache.get("c") == b"3"
    assert cache.get("d") == b"4"


def test_index_survives_reopen(tmp_path: Path) -> None:
    cache1 = AudioCache(tmp_path)
    cache1.put("persisted", b"hello")
    cache2 = AudioCache(tmp_path)
    assert cache2.get("persisted") == b"hello"


def test_invalidate_removes_entry_and_file(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path)
    cache.put("doomed", b"bye")
    assert cache.invalidate("doomed") is True
    assert cache.get("doomed") is None
    # Re-invalidate is a no-op
    assert cache.invalidate("doomed") is False


def test_clear_removes_all(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path)
    for i in range(5):
        cache.put(f"k{i}", b"x")
    cache.clear()
    assert len(cache) == 0
    for i in range(5):
        assert cache.get(f"k{i}") is None


def test_corrupted_index_recovers_gracefully(tmp_path: Path) -> None:
    cache = AudioCache(tmp_path)
    cache.put("a", b"1")
    # Corrupt the index file. Cache should treat it as empty rather than crash.
    (tmp_path / "index.json").write_text("not json at all")
    cache2 = AudioCache(tmp_path)
    assert len(cache2) == 0
    # And the cache remains usable for new entries.
    cache2.put("b", b"2")
    assert cache2.get("b") == b"2"


def test_only_known_keys_survive_reopen_when_payload_missing(tmp_path: Path) -> None:
    """If a payload file is deleted out-of-band, reopening drops that key."""
    cache = AudioCache(tmp_path)
    cache.put("ghost", b"boo")
    (tmp_path / "ghost.bin").unlink()
    cache2 = AudioCache(tmp_path)
    assert "ghost" not in cache2
    assert cache2.get("ghost") is None
