"""
D1 — Audio filtering. Post-generation only. No intent. Explicit control or fixed preset.
"""

from __future__ import annotations

from typing import List


def filter_audio_gain(pcm: List[float], gain: float = 1.0) -> List[float]:
    """
    Apply gain to PCM. Post-generation only. No intent inference.
    AC-D1-1: Applied only to post-generation output.
    AC-D1-2: gain from explicit parameter or fixed preset (default 1.0).
    AC-D1-4: Does not add notes/voices.
    AC-D1-6: Deterministic: same input + gain → same output.
    """
    if gain == 1.0:
        return list(pcm)
    return [s * gain for s in pcm]
