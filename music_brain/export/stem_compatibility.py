"""
Stem compatibility scoring.

Public API: StemCompatibilityScore.stem_pairs is Dict[Tuple[StemType, StemType], float].
Keys are always StemType pairs so callers can look up via stem_pairs[(StemType.BASS, StemType.DRUMS)].  # noqa: E501

Input stems may be passed as str or StemType; we normalize to StemType when building stem_pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, Union


class StemType(str, Enum):
    """Canonical stem role. Use for stem_pairs keys and lookups."""
    BASS = "bass"
    DRUMS = "drums"
    VOCALS = "vocals"
    OTHER = "other"

    @classmethod
    def from_key(cls, key: Union[str, "StemType"]) -> "StemType":
        """Normalize str or StemType to StemType for consistent dict keys."""
        if isinstance(key, StemType):
            return key
        s = (key or "").strip().lower()
        for st in cls:
            if st.value == s:
                return st
        return cls.OTHER


@dataclass
class StemCompatibilityScore:
    """
    Pairwise compatibility scores between stems.

    stem_pairs: (StemType, StemType) -> score in [0, 1].
    Callers should use StemType for lookups, e.g. stem_pairs[(StemType.BASS, StemType.DRUMS)].
    """
    stem_pairs: Dict[Tuple[StemType, StemType], float] = field(default_factory=dict)

    def get_score(self, a: Union[StemType, str], b: Union[StemType, str]) -> float:
        """Look up score for stem pair; accepts StemType or str. Returns 0.0 if missing."""
        ka, kb = StemType.from_key(a), StemType.from_key(b)
        key = (ka, kb) if ka.value <= kb.value else (kb, ka)
        return self.stem_pairs.get(key, 0.0)


def compute_compatibility(
    stems: Dict[Union[str, StemType], object],
) -> StemCompatibilityScore:
    """
    Build pairwise compatibility scores from a stem map.

    Input keys (stems.keys()) may be str or StemType; we normalize to StemType
    so StemCompatibilityScore.stem_pairs uses Tuple[StemType, StemType] keys.
    That way callers using stem_pairs[(StemType.BASS, StemType.DRUMS)] get correct lookups.
    """
    score = StemCompatibilityScore()
    keys = list(stems.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka = StemType.from_key(keys[i])
            kb = StemType.from_key(keys[j])
            pair = (ka, kb) if ka.value <= kb.value else (kb, ka)
            # Placeholder: real implementation would compute score from stems[keys[i]], stems[keys[j]]  # noqa: E501

            score.stem_pairs[pair] = 1.0
    return score
