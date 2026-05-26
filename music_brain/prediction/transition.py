"""First-order Markov transition model.

Deterministic baseline for sequential prediction tasks: section transitions,
chord progressions, groove patterns. ``fit`` counts (state → next_state)
occurrences across one or more sequences, and the predict / sample methods
expose the normalised transition distribution.

Used as the floor for any sequential-prediction goal in the music brain
stack — engineering above this baseline should beat it on held-out data.

Goal-list ties:
  - Predictive arrangement modeling (section sequences)
  - Future-state musical prediction (next-state distribution)
  - Harmony prediction engine (chord transition matrix)
  - Groove prediction engine (rhythmic pattern transitions)

Stdlib + numpy (for the seeded sampler).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np


class TransitionModel:
    """First-order Markov transition counts with optional Laplace smoothing.

    Parameters
    ----------
    alpha:
        Laplace pseudo-count added to every (state, next_state) pair before
        normalisation. ``alpha == 0`` gives plain maximum-likelihood
        estimates; ``alpha > 0`` assigns nonzero probability to unseen
        transitions among the observed state set.
    """

    def __init__(self, alpha: float = 0.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self._alpha = float(alpha)
        # state -> Counter({next_state: count, ...})
        self._counts: Dict[str, Counter] = {}
        self._state_set: set[str] = set()

    @property
    def states(self) -> List[str]:
        """Sorted list of observed states (sources or sinks)."""
        return sorted(self._state_set)

    def fit(self, sequences: Iterable[Sequence[str]]) -> "TransitionModel":
        """Count transitions across one or more sequences.

        ``fit`` is additive — calling it multiple times accumulates counts
        rather than resetting. ``reset()`` wipes the model if needed.
        """
        for seq in sequences:
            for i, state in enumerate(seq):
                self._state_set.add(state)
                if i + 1 >= len(seq):
                    continue
                nxt = seq[i + 1]
                self._state_set.add(nxt)
                self._counts.setdefault(state, Counter())[nxt] += 1
        return self

    def reset(self) -> None:
        """Drop all accumulated counts."""
        self._counts.clear()
        self._state_set.clear()

    def predict_next(self, state: str) -> Dict[str, float]:
        """Return the next-state probability distribution given ``state``.

        - Returns ``{}`` for states never observed at all.
        - Returns ``{}`` for states observed only as a terminal (no outgoing
          edges) unless ``alpha > 0``, in which case smoothing puts mass on
          every observed state.
        - With ``alpha > 0``, the support is the full observed state set.
        """
        if state not in self._state_set:
            return {}
        outgoing = self._counts.get(state, Counter())
        if not outgoing and self._alpha == 0.0:
            return {}
        # With smoothing, candidate set is all observed states.
        candidates = self._state_set if self._alpha > 0 else set(outgoing)
        total = sum(outgoing.values()) + self._alpha * len(candidates)
        if total == 0:
            return {}
        return {
            s: (outgoing.get(s, 0) + self._alpha) / total
            for s in candidates
        }

    def sample_next(self, state: str, rng: np.random.Generator) -> str:
        """Sample one next-state from ``predict_next(state)``.

        Raises ``KeyError`` when ``state`` has no observed transitions and
        smoothing is disabled.
        """
        dist = self.predict_next(state)
        if not dist:
            raise KeyError(f"no outgoing transitions from state {state!r}")
        keys = list(dist.keys())
        probs = np.array([dist[k] for k in keys], dtype=np.float64)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(keys), p=probs))
        return keys[idx]

    def predict_sequence(
        self,
        start: str,
        length: int,
        rng: np.random.Generator,
    ) -> List[str]:
        """Sample a Markov sequence of ``length`` states starting from ``start``.

        Returns ``[start, ...]``. Terminates early if the chain reaches a
        state with no outgoing transitions (and no smoothing to fill in).
        """
        if length <= 0:
            return []
        out = [start]
        current = start
        while len(out) < length:
            try:
                current = self.sample_next(current, rng)
            except KeyError:
                break
            out.append(current)
        return out
