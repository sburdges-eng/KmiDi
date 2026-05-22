"""File-backed LRU cache for audio analysis artifacts.

Caches deterministic outputs of audio analysis (spectrograms, embeddings,
feature vectors) keyed by a content hash of the source audio plus the
analyzer's version string. Cache hits skip the analyzer entirely.

Scope (per docs/ARCHITECTURAL_CONTRACTS.md §17 deterministic boundary):
- Cache deterministic outputs only. Probabilistic outputs (sampled tokens,
  stochastic generations) must NOT be cached here.
- File-backed, not in-memory. Survives process restart.
- Single-process safe; not multi-process safe (no file locking). Cross-process
  cooperation is intentionally out of scope.

Storage layout under ``cache_dir``::

    <cache_dir>/
        index.json           # ordered key list, oldest first (LRU eviction order)
        <hash>.bin           # cached payload (numpy .npy or raw bytes)

Index format::

    {
        "version": 1,
        "max_entries": 256,
        "keys": ["abc123...", "def456...", ...]   # MRU is last
    }
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

_INDEX_VERSION = 1
_DEFAULT_MAX_ENTRIES = 256


def _hash_key(source_bytes: bytes, analyzer_id: str) -> str:
    """Stable cache key from raw audio bytes + analyzer identifier."""
    h = hashlib.sha256()
    h.update(analyzer_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(source_bytes)
    return h.hexdigest()


class AudioCache:
    """File-backed LRU cache for deterministic audio analysis outputs.

    Args:
        cache_dir: Directory to hold the index and payloads. Created if
            missing.
        max_entries: Soft cap on cache size. When exceeded, the oldest
            entries are evicted.
    """

    def __init__(self, cache_dir: Path, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = int(max_entries)
        self._index_path = self._dir / "index.json"
        self._keys: "OrderedDict[str, None]" = OrderedDict()
        self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def key_for(self, source_bytes: bytes, analyzer_id: str) -> str:
        """Compute the cache key the analyzer would use for these inputs."""
        return _hash_key(source_bytes, analyzer_id)

    def get(self, key: str) -> Optional[bytes]:
        """Return the cached payload, marking it MRU; None if absent."""
        path = self._payload_path(key)
        if not path.exists():
            self._keys.pop(key, None)
            return None
        # Mark MRU by moving to end of the ordered dict.
        if key in self._keys:
            self._keys.move_to_end(key)
        else:
            self._keys[key] = None
        self._save_index()
        return path.read_bytes()

    def put(self, key: str, payload: bytes) -> None:
        """Store ``payload`` under ``key``. Triggers eviction if over cap."""
        path = self._payload_path(key)
        path.write_bytes(payload)
        if key in self._keys:
            self._keys.move_to_end(key)
        else:
            self._keys[key] = None
        self._evict_if_needed()
        self._save_index()

    def invalidate(self, key: str) -> bool:
        """Remove a single entry. Returns True if it existed."""
        had = key in self._keys
        self._keys.pop(key, None)
        path = self._payload_path(key)
        if path.exists():
            path.unlink()
        self._save_index()
        return had

    def clear(self) -> None:
        """Remove all entries (does not delete the cache directory itself)."""
        for key in list(self._keys.keys()):
            path = self._payload_path(key)
            if path.exists():
                path.unlink()
        self._keys.clear()
        self._save_index()

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._keys

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _payload_path(self, key: str) -> Path:
        return self._dir / f"{key}.bin"

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text())
        except (OSError, json.JSONDecodeError):
            # Corrupted index — treat as empty rather than crash. Existing
            # payload files become orphans that the next clear() will remove.
            self._keys.clear()
            return
        if not isinstance(data, dict) or data.get("version") != _INDEX_VERSION:
            self._keys.clear()
            return
        for k in data.get("keys", []):
            if isinstance(k, str) and self._payload_path(k).exists():
                self._keys[k] = None

    def _save_index(self) -> None:
        data = {
            "version": _INDEX_VERSION,
            "max_entries": self._max,
            "keys": list(self._keys.keys()),
        }
        self._index_path.write_text(json.dumps(data))

    def _evict_if_needed(self) -> None:
        while len(self._keys) > self._max:
            oldest_key, _ = self._keys.popitem(last=False)
            path = self._payload_path(oldest_key)
            if path.exists():
                path.unlink()


__all__ = ["AudioCache"]
