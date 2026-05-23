"""Semantic motif persistence + recurrence tracking.

Closes **Semantic motif persistence**, **Motif recurrence tracking**,
and provides the substrate for **Arrangement continuity preservation**.

A motif is an embedded latent fingerprint — the mean-pooled
``audio_z`` of a ``LatentFrame``. The tracker maintains a small list
of canonical motif representatives, computes cosine similarity for
each new observation, and either:

  - records the observation against the closest existing motif if
    similarity ≥ ``similarity_threshold``, or
  - creates a new motif entry otherwise.

This is a small, in-memory, *online* clusterer (no offline batch
re-fitting), which fits the realtime composer workflow: each new
phrase the AI emits either echoes a motif (continuity) or introduces
one (variety).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from music_brain.latent.latent_frame import LatentFrame


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class MotifRecurrence:
    motif_id: int
    is_new: bool
    occurrences: int
    time_indices: Tuple[int, ...]
    representative: object  # torch.Tensor; opaque type so doc gen stays clean


class _MotifEntry:
    __slots__ = ("motif_id", "representative", "time_indices")

    def __init__(self, motif_id: int, representative, time_index: int) -> None:
        self.motif_id = motif_id
        self.representative = representative
        self.time_indices: list[int] = [time_index]


class MotifTracker:
    """Online cosine-similarity motif clusterer.

    Args:
        similarity_threshold: minimum cosine similarity for an
            observation to be classed as a recurrence of an existing
            motif. Range [0, 1].
    """

    def __init__(self, similarity_threshold: float = 0.92) -> None:
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
        self._threshold = float(similarity_threshold)
        self._motifs: list[_MotifEntry] = []
        self._dim: Optional[int] = None
        self._next_id = 0

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def motif_count(self) -> int:
        return len(self._motifs)

    def most_recurrent(self) -> Optional[MotifRecurrence]:
        if not self._motifs:
            return None
        winner = max(self._motifs, key=lambda m: len(m.time_indices))
        return MotifRecurrence(
            motif_id=winner.motif_id,
            is_new=False,
            occurrences=len(winner.time_indices),
            time_indices=tuple(winner.time_indices),
            representative=winner.representative,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, frame: LatentFrame) -> MotifRecurrence:
        torch = _torch()
        pooled = frame.audio_z.mean(dim=0)  # (D,)
        if self._dim is None:
            self._dim = int(pooled.shape[0])
        elif pooled.shape[0] != self._dim:
            raise ValueError(f"motif dim {pooled.shape[0]} != tracker dim {self._dim}")
        norm = float(pooled.norm())
        if norm < 1e-12:
            unit = pooled
        else:
            unit = pooled / norm

        best_idx = -1
        best_score = -1.0
        for i, motif in enumerate(self._motifs):
            rep_norm = float(motif.representative.norm())
            if rep_norm < 1e-12:
                continue
            sim = float(torch.dot(unit, motif.representative / rep_norm))
            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_idx >= 0 and best_score >= self._threshold and norm >= 1e-12:
            motif = self._motifs[best_idx]
            motif.time_indices.append(frame.time_index)
            return MotifRecurrence(
                motif_id=motif.motif_id,
                is_new=False,
                occurrences=len(motif.time_indices),
                time_indices=tuple(motif.time_indices),
                representative=motif.representative,
            )

        # New motif.
        new_id = self._next_id
        self._next_id += 1
        entry = _MotifEntry(motif_id=new_id, representative=pooled, time_index=frame.time_index)
        self._motifs.append(entry)
        return MotifRecurrence(
            motif_id=new_id,
            is_new=True,
            occurrences=1,
            time_indices=(frame.time_index,),
            representative=pooled,
        )


__all__ = ["MotifTracker", "MotifRecurrence"]
