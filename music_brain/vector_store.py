"""Numpy-backed vector store with persistence.

Closes two spec items in one module:
  - Vector memory retrieval
  - Embedding database persistence

Scope:
- In-memory store with cosine-similarity top-k search.
- Single-process safe; no concurrent-writer protection.
- File-backed persistence via numpy's compressed savez format.
- Stable IDs (caller-supplied strings) decoupled from internal row indices,
  so deletions don't require renumbering external references.

Not a replacement for FAISS / Chroma / pgvector. This is the local-first
default that ships in-tree so retrieval-augmented features have a working
baseline without an external service dependency. When a heavier vector
backend is needed, the same public API can be reused by a subclass.

Persistence layout (``save(path)`` writes a single ``.npz`` with):
  - vectors:  (N, D) float32 array, L2-normalised at insert time
  - ids:      (N,) object array (UTF-8 strings)
  - meta:     (N,) object array (JSON-encoded metadata or empty string)
  - dim:      scalar int (embedding dimension)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def _normalise(v: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of ``v`` (handles zero vectors safely)."""
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return v
    return v / norm


class VectorStore:
    """In-memory cosine-similarity vector store with .npz persistence.

    Args:
        dim: Embedding dimension. Must match every vector added.

    Vectors are L2-normalised on insert so search reduces to a single
    matrix multiply with no per-query normalisation cost.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = int(dim)
        self._vectors = np.zeros((0, self._dim), dtype=np.float32)
        self._ids: list[str] = []
        self._meta: list[Optional[dict]] = []
        # Reverse index: id -> row. Rebuilt on deletes for simplicity.
        self._index: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._index

    def add(self, id_: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None:
        """Insert or replace a vector under ``id_``.

        Replacing an existing id preserves the row index so external
        references stay stable.
        """
        if not isinstance(id_, str) or not id_:
            raise ValueError("id_ must be a non-empty string")
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self._dim:
            raise ValueError(f"vector dim {vec.shape[0]} != store dim {self._dim}")
        normed = _normalise(vec)
        if id_ in self._index:
            row = self._index[id_]
            self._vectors[row] = normed
            self._meta[row] = metadata
            return
        self._vectors = np.vstack([self._vectors, normed[np.newaxis, :]])
        self._ids.append(id_)
        self._meta.append(metadata)
        self._index[id_] = len(self._ids) - 1

    def get(self, id_: str) -> Optional[tuple[np.ndarray, Optional[dict]]]:
        """Return ``(vector, metadata)`` for ``id_`` or None if absent."""
        row = self._index.get(id_)
        if row is None:
            return None
        return self._vectors[row].copy(), self._meta[row]

    def delete(self, id_: str) -> bool:
        """Remove ``id_``. Returns True if it existed."""
        row = self._index.pop(id_, None)
        if row is None:
            return False
        self._vectors = np.delete(self._vectors, row, axis=0)
        self._ids.pop(row)
        self._meta.pop(row)
        # Rebuild reverse index (row indices shifted after delete).
        self._index = {k: i for i, k in enumerate(self._ids)}
        return True

    def search(
        self, query: np.ndarray, top_k: int = 5, min_score: float = -1.0
    ) -> list[tuple[str, float, Optional[dict]]]:
        """Return up to ``top_k`` nearest neighbours by cosine similarity.

        Args:
            query: (D,) embedding to search against.
            top_k: maximum results to return.
            min_score: discard hits with cosine < this threshold.

        Returns:
            Ranked list of ``(id, score, metadata)`` tuples, highest score
            first. Scores are in [-1, 1].
        """
        if len(self._ids) == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError(f"query dim {q.shape[0]} != store dim {self._dim}")
        q = _normalise(q)
        # Cosine sim = dot product because both sides are unit-length.
        scores = self._vectors @ q
        # Argpartition for O(N) top-k, then sort the small slice.
        k = min(int(top_k), scores.shape[0])
        if k <= 0:
            return []
        if k == scores.shape[0]:
            top_idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k - 1)[:k]
            top_idx = part[np.argsort(-scores[part])]
        out: list[tuple[str, float, Optional[dict]]] = []
        for row in top_idx:
            s = float(scores[int(row)])
            if s < min_score:
                continue
            out.append((self._ids[int(row)], s, self._meta[int(row)]))
        return out

    def save(self, path: Path) -> None:
        """Serialise the store to a single .npz file at ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_blobs = np.asarray(
            [json.dumps(m) if m is not None else "" for m in self._meta], dtype=object
        )
        np.savez_compressed(
            path,
            vectors=self._vectors.astype(np.float32),
            ids=np.asarray(self._ids, dtype=object),
            meta=meta_blobs,
            dim=np.asarray(self._dim),
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Load a store previously written by ``save``."""
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            dim = int(data["dim"])
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            ids = [str(x) for x in data["ids"].tolist()]
            meta_raw = data["meta"].tolist()
        store = cls(dim)
        if vectors.shape[0] > 0:
            if vectors.shape[1] != dim:
                raise ValueError(f"saved vectors dim {vectors.shape[1]} != saved dim {dim}")
            store._vectors = vectors
            store._ids = ids
            store._meta = [json.loads(m) if m else None for m in meta_raw]
            store._index = {k: i for i, k in enumerate(store._ids)}
        return store


__all__ = ["VectorStore"]
