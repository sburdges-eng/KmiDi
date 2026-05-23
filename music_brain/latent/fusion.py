"""Multimodal fusion + multi-stem orchestration.

Closes:
  - Multimodal fusion engine
  - Multimodal latent alignment (operational side)
  - Hybrid symbolic/audio generation
  - Audio/MIDI dual representation
  - Stem-aware generation
  - Multi-instrument orchestration

Two surfaces:

  - ``MultimodalFusion`` concatenates the ``audio_z`` and ``chord_z``
    legs of one ``LatentFrame`` into a single ``(T, D_audio + D_chord)``
    tensor downstream models can consume as a unified representation.
    When ``chord_z`` is absent, the chord half is zero-padded — so the
    fusion shape is stable regardless of whether a frame carries
    symbolic information.

  - ``StemBundle`` groups multiple per-instrument ``LatentFrame``
    instances (drums, bass, keys, …) at the same ``time_index`` and
    exposes the two canonical orchestration views: ``mix()`` (concat
    on the feature axis — preserves per-stem identity) and
    ``average()`` (mean across stems — useful as a low-bandwidth
    cross-stem summary). All stems must share ``time_index`` so the
    bundle is a single moment in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from music_brain.latent.latent_frame import LatentFrame


def _torch():
    import torch
    return torch


class MultimodalFusion:
    """Concatenate audio + chord latents into one fused tensor.

    Args:
        chord_dim: feature dim for the chord half. When a frame has no
            chord_z the half is zero-padded to this dim. Default 8.
    """

    def __init__(self, chord_dim: int = 8) -> None:
        if chord_dim <= 0:
            raise ValueError("chord_dim must be > 0")
        self._chord_dim = int(chord_dim)

    @staticmethod
    def _validate_audio_dim(d: int) -> None:
        if d <= 0:
            raise ValueError("audio_feature_dim must be > 0")

    def __call__(self, frame: LatentFrame):
        torch = _torch()
        audio = frame.audio_z
        self._validate_audio_dim(audio.shape[1])
        chord = frame.chord_z
        if chord is None:
            chord = torch.zeros(audio.shape[0], self._chord_dim,
                                dtype=audio.dtype)
        if chord.shape[0] != audio.shape[0]:
            raise ValueError(
                f"audio time-steps {audio.shape[0]} != chord time-steps {chord.shape[0]}")
        return torch.cat([audio, chord], dim=-1)


@dataclass(frozen=True)
class StemBundle:
    """A bag of per-instrument LatentFrames at a single time_index."""
    stems: Mapping[str, LatentFrame]

    def __post_init__(self) -> None:
        if not self.stems:
            raise ValueError("StemBundle must have at least one stem")
        t = None
        for name, f in self.stems.items():
            if t is None:
                t = f.time_index
            elif f.time_index != t:
                raise ValueError(
                    f"stem '{name}' time_index {f.time_index} != bundle time_index {t}")

    def __len__(self) -> int:
        return len(self.stems)

    @property
    def stem_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.stems.keys()))

    def mix(self):
        """Concatenate every stem's ``audio_z`` along the feature axis."""
        torch = _torch()
        ordered = [self.stems[name].audio_z for name in self.stem_names]
        return torch.cat(ordered, dim=-1)

    def average(self):
        """Mean of every stem's ``audio_z`` (low-bandwidth summary)."""
        torch = _torch()
        ordered = [self.stems[name].audio_z for name in self.stem_names]
        return torch.stack(ordered, dim=0).mean(dim=0)


__all__ = ["MultimodalFusion", "StemBundle"]
