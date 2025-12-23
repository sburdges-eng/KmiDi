"""
Sample utilities for music_brain.

Exposes emotion-scale sampling helpers and API fetchers.
"""

from music_brain.samples.emotion_scale_sampler import (
    EmotionScaleSampler,
    FreesoundFetcher,
)

__all__ = [
    "EmotionScaleSampler",
    "FreesoundFetcher",
]

