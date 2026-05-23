"""Shared latent contract for cross-encoder composition.

The KmiDi latent stack (``music_brain/cross_attention.py``,
``music_brain/world_model.py``, ``music_brain/vector_store.py``,
``music_brain/jepa/{audio_jepa,chord_jepa,emotion_probe}.py``) emits
heterogeneous tensors with no shared invariants. Without a common
envelope, downstream goals — multimodal latent alignment, persistent
latent world-state, chunked latent prediction, conditioning-bridge
composition — cannot be wired end-to-end.

``LatentFrame`` is that envelope. Each frame represents one *unbatched*
window of latent state with:

  - ``audio_z``: (T, D_audio) float32 — JEPA-style audio latents.
  - ``chord_z``: optional (T_c, D_chord) float32 — chord-JEPA latents.
  - ``emotion_va``: (valence, arousal) in [-1, 1] — emotion-probe
    output; carried as Python floats so a frame remains useful even
    when only a pooled VA estimate is available.
  - ``time_index``: monotonic chunk index. Mirrors
    ``StreamingWavWriter.chunk_index`` so deterministic-ordering
    consumers can interleave audio and latent streams without a side
    channel.
  - ``provenance``: reused from ``music_brain/intent_ir/__init__.py``.
    Provenance is essential for the closed-loop personalization PR
    (PR3 in the plan) where retrieval must distinguish user-authored
    intent from ML-inferred intent.
  - ``metadata``: read-only key/value bag for diagnostics. Wrapped in
    ``MappingProxyType`` so callers can't mutate a frame's metadata
    through a back-channel after construction.

The contract is *minimal on purpose*: anything beyond what's needed to
compose existing modules is a future PR.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from music_brain.intent_ir import IntentProvenance, IntentSource


def _torch():
    import torch
    return torch


_FLOAT32_DTYPE_NAME = "torch.float32"


def _check_latent_tensor(name: str, t, *, expected_feature_dim: Optional[int] = None) -> None:
    torch = _torch()
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    if t.dim() != 2:
        raise ValueError(
            f"{name} must be a 2-D (T, D) tensor; got shape {tuple(t.shape)}")
    if t.dtype != torch.float32:
        raise ValueError(
            f"{name} dtype must be torch.float32; got {t.dtype}")
    if expected_feature_dim is not None and t.shape[1] != expected_feature_dim:
        raise ValueError(
            f"{name} feature dim {t.shape[1]} != expected {expected_feature_dim}")


def _freeze_metadata(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    # Copy first so caller-side mutation can't reach back.
    return MappingProxyType(dict(metadata))


@dataclass(frozen=True)
class LatentFrame:
    """Immutable cross-encoder latent envelope.

    All tensors are *unbatched* (no leading batch dim). Construct one
    frame per sample / time-window. Use ``advance(...)`` to produce a
    chained successor frame with an incremented ``time_index``.
    """

    audio_z: Any  # torch.Tensor (T, D_audio)
    chord_z: Optional[Any]  # torch.Tensor (T_c, D_chord) or None
    emotion_va: Tuple[float, float]
    time_index: int
    provenance: IntentProvenance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_latent_tensor("audio_z", self.audio_z)
        if self.chord_z is not None:
            _check_latent_tensor("chord_z", self.chord_z)
        if not (isinstance(self.emotion_va, tuple) and len(self.emotion_va) == 2):
            raise ValueError("emotion_va must be a (valence, arousal) tuple")
        v, a = self.emotion_va
        if not (-1.0 <= float(v) <= 1.0) or not (-1.0 <= float(a) <= 1.0):
            raise ValueError("emotion_va components must be in [-1, 1]")
        if not isinstance(self.time_index, int) or self.time_index < 0:
            raise ValueError("time_index must be a non-negative int")
        if not isinstance(self.provenance, IntentProvenance):
            raise TypeError("provenance must be an IntentProvenance instance")
        # Re-freeze metadata even if a caller passed a plain dict — frozen
        # dataclass means we go through object.__setattr__.
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        # Coerce emotion_va components to plain floats for byte-stable serialization.
        object.__setattr__(
            self, "emotion_va", (float(self.emotion_va[0]), float(self.emotion_va[1])))

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @property
    def audio_feature_dim(self) -> int:
        return int(self.audio_z.shape[1])

    @property
    def audio_time_steps(self) -> int:
        return int(self.audio_z.shape[0])

    # ------------------------------------------------------------------
    # Chained successor
    # ------------------------------------------------------------------

    def advance(self, *, time_index: int, audio_z,
                chord_z=None,
                emotion_va: Optional[Tuple[float, float]] = None,
                metadata: Optional[Mapping[str, Any]] = None) -> "LatentFrame":
        """Return a successor frame at ``time_index``.

        Enforces strict-monotonic time and feature-dim continuity.
        ``provenance`` propagates by default; pass a new ``metadata``
        to override the diagnostics bag.
        """
        if time_index <= self.time_index:
            raise ValueError(
                f"time_index must be strictly monotonic; "
                f"got {time_index} <= current {self.time_index}")
        _check_latent_tensor(
            "audio_z", audio_z,
            expected_feature_dim=self.audio_feature_dim)
        if chord_z is not None and self.chord_z is not None:
            _check_latent_tensor(
                "chord_z", chord_z,
                expected_feature_dim=int(self.chord_z.shape[1]))
        return replace(
            self,
            audio_z=audio_z,
            chord_z=chord_z if chord_z is not None else (
                self.chord_z.to(audio_z.device)
                if self.chord_z is not None else None),
            emotion_va=emotion_va if emotion_va is not None else self.emotion_va,
            time_index=time_index,
            metadata=metadata if metadata is not None else dict(self.metadata),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation.

        Tensors are stored as nested float lists. This is intentionally
        simple — the goal is round-trip fidelity for unit tests and
        cross-process IPC at modest sizes, not a high-throughput on-disk
        format. Use ``vector_store.save`` for bulk persistence.
        """
        return {
            "audio_z": self.audio_z.detach().cpu().tolist(),
            "audio_dtype": _FLOAT32_DTYPE_NAME,
            "chord_z": (self.chord_z.detach().cpu().tolist()
                        if self.chord_z is not None else None),
            "emotion_va": list(self.emotion_va),
            "time_index": int(self.time_index),
            "provenance": {
                "source": int(self.provenance.source),
                "user_override_weight": float(self.provenance.user_override_weight),
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentFrame":
        torch = _torch()
        audio_z = torch.tensor(payload["audio_z"], dtype=torch.float32)
        chord_payload = payload.get("chord_z")
        chord_z = (torch.tensor(chord_payload, dtype=torch.float32)
                   if chord_payload is not None else None)
        prov_payload = payload["provenance"]
        provenance = IntentProvenance(
            source=IntentSource(int(prov_payload["source"])),
            user_override_weight=float(prov_payload["user_override_weight"]),
        )
        emo = payload["emotion_va"]
        return cls(
            audio_z=audio_z,
            chord_z=chord_z,
            emotion_va=(float(emo[0]), float(emo[1])),
            time_index=int(payload["time_index"]),
            provenance=provenance,
            metadata=dict(payload.get("metadata", {})),
        )


__all__ = ["LatentFrame"]
