"""Retrieval-augmented generation memory over ``LatentFrame``.

Closes a cluster of spec items in one module:

  - Vector memory retrieval
  - Retrieval-augmented generation
  - Emotion-conditioned retrieval
  - Embedding-indexed user history
  - Semantic retrieval steering
  - Context persistence beyond token window

Builds on the existing in-tree ``music_brain/vector_store.VectorStore``
rather than introducing a new backend. The closed-loop personalization
PR called for a sqlite-vec swap; we keep that as a *future* drop-in
(VectorStore was designed with a swappable subclass surface) and ship
the contract today.

A ``LatentMemory`` instance:

  - Embeds a frame by mean-pooling its ``audio_z`` along the time
    axis.
  - Stores the pooled vector keyed by a caller-supplied id, with
    metadata that propagates ``emotion_va``, ``time_index``,
    ``provenance``, and arbitrary domain fields (``user_id``,
    ``satisfaction``, …).
  - Recalls top-k entries by cosine similarity, optionally weighted by
    proximity in (valence, arousal) space.
  - Filters by ``user_id`` so the longitudinal user model can scope
    retrieval to one person at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from music_brain.latent.latent_frame import LatentFrame
from music_brain.vector_store import VectorStore


def pool_audio_z(audio_z) -> Any:
    """Mean-pool an unbatched ``(T, D)`` latent tensor down to ``(D,)``."""
    if audio_z.dim() != 2:
        raise ValueError(f"pool_audio_z expects 2-D (T, D); got shape {tuple(audio_z.shape)}")
    return audio_z.mean(dim=0)


@dataclass(frozen=True)
class RecallHit:
    id: str
    score: float
    metadata: Mapping[str, Any]


def _emotion_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dv = a[0] - b[0]
    da = a[1] - b[1]
    return float((dv * dv + da * da) ** 0.5)


def _emotion_proximity_score(query_va, stored_va) -> float:
    """Map VA distance to a [0, 1] proximity score.

    Max distance in [-1, 1] x [-1, 1] is 2*sqrt(2) ≈ 2.828. We map
    0 → 1.0 (perfect match), full-distance → 0.0 (opposite).
    """
    max_dist = 2.828427124746
    d = _emotion_distance(query_va, stored_va)
    return max(0.0, 1.0 - d / max_dist)


class LatentMemory:
    """Embedded ``LatentFrame`` memory with emotion-conditioned recall."""

    def __init__(self, dim: int) -> None:
        self._dim = int(dim)
        self._store = VectorStore(dim=self._dim)
        # Sidecar so we can slice by user without scanning every metadata.
        self._user_index: dict[str, set[str]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def remember(
        self, id_: str, frame: LatentFrame, *, metadata: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Insert (or replace) one frame's pooled embedding."""
        pooled = pool_audio_z(frame.audio_z).detach().cpu().numpy()
        if pooled.shape[0] != self._dim:
            raise ValueError(f"frame audio_feature_dim {pooled.shape[0]} != memory dim {self._dim}")
        meta = {
            "time_index": frame.time_index,
            "emotion_va": list(frame.emotion_va),
            "provenance_source": int(frame.provenance.source),
            "user_override_weight": float(frame.provenance.user_override_weight),
        }
        if metadata:
            meta.update(dict(metadata))
        self._store.add(id_, pooled.astype(np.float32), metadata=meta)
        user_id = meta.get("user_id")
        if isinstance(user_id, str):
            self._user_index.setdefault(user_id, set()).add(id_)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query_frame: LatentFrame,
        *,
        top_k: int = 5,
        emotion_weight: float = 0.0,
        user_id: Optional[str] = None,
    ) -> list[RecallHit]:
        if not (0.0 <= emotion_weight <= 1.0):
            raise ValueError(f"emotion_weight must be in [0, 1], got {emotion_weight}")
        if len(self._store) == 0:
            return []
        pooled = pool_audio_z(query_frame.audio_z).detach().cpu().numpy()

        if user_id is not None:
            # User-scoped path: score *only* this user's vectors directly.
            # The previous "over-fetch top_k*4 globally then post-filter"
            # approach silently returned zero hits when other users
            # dominated the nearest neighbors. Direct scoring of the
            # user's own index is O(M_user) and always returns top_k as
            # long as the user has that many vectors.
            candidate_ids = self._user_index.get(user_id, set())
            return self._recall_within(
                pooled.astype(np.float32),
                candidate_ids,
                query_va=query_frame.emotion_va,
                top_k=top_k,
                emotion_weight=emotion_weight,
            )

        # Unscoped path: existing fast global top-k search.
        raw_hits = self._store.search(pooled.astype(np.float32), top_k=top_k)
        hits: list[RecallHit] = []
        for hid, score, meta in raw_hits:
            stored_va = tuple(meta.get("emotion_va", [0.0, 0.0])) if meta else (0.0, 0.0)
            if emotion_weight > 0.0:
                emo_prox = _emotion_proximity_score(
                    query_frame.emotion_va, (float(stored_va[0]), float(stored_va[1]))
                )
                blended = (1.0 - emotion_weight) * score + emotion_weight * emo_prox
            else:
                blended = score
            hits.append(RecallHit(id=hid, score=float(blended), metadata=meta or {}))
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    def _recall_within(
        self,
        query_vec: np.ndarray,
        candidate_ids,
        *,
        query_va,
        top_k: int,
        emotion_weight: float,
    ) -> list[RecallHit]:
        """Score a fixed candidate-id set against ``query_vec``.

        Used by ``recall(user_id=...)`` so the search domain is the
        user's own indexed vectors rather than the global top-k tail.
        """
        if not candidate_ids:
            return []
        # Normalise the query once so dot product == cosine sim against
        # the L2-normalised vectors the store holds.
        q = query_vec.reshape(-1).astype(np.float32)
        qn = float(np.linalg.norm(q))
        if qn >= 1e-12:
            q = q / qn
        hits: list[RecallHit] = []
        for cid in candidate_ids:
            got = self._store.get(cid)
            if got is None:
                continue
            vec, meta = got
            score = float(np.dot(vec, q))
            stored_va = tuple(meta.get("emotion_va", [0.0, 0.0])) if meta else (0.0, 0.0)
            if emotion_weight > 0.0:
                emo_prox = _emotion_proximity_score(
                    query_va, (float(stored_va[0]), float(stored_va[1]))
                )
                blended = (1.0 - emotion_weight) * score + emotion_weight * emo_prox
            else:
                blended = score
            hits.append(RecallHit(id=cid, score=float(blended), metadata=meta or {}))
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        self._store.save(path)

    @classmethod
    def load(cls, path: Path) -> "LatentMemory":
        store = VectorStore.load(path)
        mem = cls(dim=store.dim)
        mem._store = store
        # Rebuild user index by reading rows directly from the store.
        # The previous implementation rebuilt via ``store.search(row, top_k=1)``,
        # which returns *some* row whose embedding matches — when several
        # rows share the same normalised embedding (a frequent case for
        # duplicated motif frames), top-1 hops to a different row's id and
        # the original row never joins the index.
        for row_id, meta in zip(store._ids, store._meta):  # type: ignore[attr-defined]
            if not meta:
                continue
            user_id = meta.get("user_id")
            if isinstance(user_id, str):
                mem._user_index.setdefault(user_id, set()).add(row_id)
        return mem

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot_vectors(self) -> np.ndarray:
        return self._store._vectors.copy()  # type: ignore[attr-defined]


__all__ = ["LatentMemory", "RecallHit", "pool_audio_z"]
