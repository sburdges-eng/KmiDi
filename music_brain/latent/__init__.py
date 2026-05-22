"""Latent-space utilities: fusion, retrieval, projection."""

from music_brain.latent.fusion import (
    concat_fuse,
    gated_fuse,
    normalized_average,
    weighted_sum,
)

__all__ = [
    "concat_fuse",
    "gated_fuse",
    "normalized_average",
    "weighted_sum",
]
