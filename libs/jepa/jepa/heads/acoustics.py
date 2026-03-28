"""
Acoustics head for JEPA: project context/target embeddings to SPL or response curve.

Used by SubWooder and audio-enclosure pipelines. Stub: implement forward pass
to map JEPA embeddings to acoustic targets (e.g. frequency response, SPL).
"""
from typing import Any


def acoustics_head_forward(embedding: Any, *, freq_bins: int = 128) -> Any:
    """Map JEPA embedding to acoustic output (e.g. magnitude curve). Stub: raise until implemented."""
    raise NotImplementedError("JEPA acoustics head not implemented")
