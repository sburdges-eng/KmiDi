"""Intent → cross-attention conditioning bridge.

Closes the **Conditioning bridge projection layer** spec item by
mapping an ``IntentFrame`` (the canonical cross-language intent unit
defined at ``shared_schemas/intent_frame_schema.json`` and mirrored in
``music_brain/intent_ir/__init__.py``) into a ``(B, S_ctx, kv_dim)``
tensor consumable as the ``kv`` argument of
``music_brain/cross_attention.py:CrossAttention.forward``.

Why "slots" instead of one big concatenated vector:

The cross-attention layer is *the* place where downstream model
weights learn to weigh competing facets of the intent (emotion vs.
musical preference vs. timing window vs. provenance/constraints). If
we collapse everything into one token, the attention layer has nothing
to attend over — softmax over a single key is the identity. By
materializing one slot per facet (4 slots), the attention weights
become inspectable: each query position emits a 4-vector saying "I
listened 60% to emotion, 30% to musical preference, …" — which is
both the right inductive bias and the right diagnostic surface.

The featurizer is *deterministic and total*: every ``IntentFrame``
field round-trips into a well-conditioned float (sentinels like
``start_bar == -1`` and ``max_event_rate == inf`` are pre-normalized so
no NaN/inf ever reaches the trainable Linear). The Linear layers
themselves are seeded by the standard PyTorch init, so call
``torch.manual_seed(...)`` before constructing the bridge if byte-
identical weights matter.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

from music_brain.intent_ir import IntentFrame


def _torch():
    import torch
    return torch


# Public stability: number of intent "tokens" the bridge emits.
INTENT_SLOTS = 4

_EMOTION_DIM = 4
_MUSIC_DIM = 10
_TIME_DIM = 4
_PROVENANCE_DIM = 4


def _featurize_emotion(frame: IntentFrame) -> list[float]:
    e = frame.emotion
    # All four are already documented in [-1, 1] or [0, 1] in IR v1.
    return [float(e.valence), float(e.arousal), float(e.dominance),
            float(e.confidence)]


def _featurize_music(frame: IntentFrame) -> list[float]:
    m = frame.music
    # mode_preference is the only signed int in the bunch; normalize to
    # match the [-1, 1] range of tempo_bias for a well-conditioned input.
    return [
        float(m.tempo_bias),
        float(m.rhythmic_density),
        float(m.groove_strength),
        float(m.harmonic_tension),
        float(m.harmonic_motion),
        float(m.mode_preference),  # in {-1, 0, +1}
        float(m.melodic_activity),
        float(m.contour_variance),
        float(m.dynamic_range),
        float(m.texture_density),
    ]


def _normalize_bar(bar: int) -> float:
    """Map a bar index to a well-conditioned float.

    ``-1`` is the "immediate / open-ended" sentinel in IR v1. We map it
    to 0.0 and otherwise scale linearly by 1/32 so a typical 8-bar
    section sits at 0.25 — comfortable for a fresh Linear init.
    """
    if bar < 0:
        return 0.0
    return float(bar) / 32.0


def _featurize_time(frame: IntentFrame) -> list[float]:
    t = frame.time
    return [
        _normalize_bar(int(t.start_bar)),
        _normalize_bar(int(t.end_bar)),
        float(t.fade_in_beats) / 16.0,
        float(t.fade_out_beats) / 16.0,
    ]


def _featurize_provenance(frame: IntentFrame) -> list[float]:
    p = frame.provenance
    c = frame.constraints
    # IntentSource values are 0..5; normalize to [0, 1] so a fresh
    # Linear sees a unit-scale input. max_event_rate defaults to +inf;
    # squash through a bounded transform.
    import math
    max_evt = float(c.max_event_rate)
    if math.isinf(max_evt) or math.isnan(max_evt):
        max_evt_norm = 1.0
    else:
        # Map [0, inf) → [0, 1) without an asymptote-jump near zero.
        max_evt_norm = math.tanh(max_evt / 1000.0)
    return [
        float(int(p.source)) / 5.0,
        float(p.user_override_weight),
        float(c.max_cpu_cost),
        max_evt_norm,
    ]


class ConditioningProjection:
    """Map ``IntentFrame`` → cross-attention key/value tokens.

    Args:
        kv_dim: feature dimension of each emitted intent slot. Must
            match the ``kv_dim`` (or ``query_dim`` when symmetric) of
            the downstream ``CrossAttention``.
    """

    def __init__(self, kv_dim: int) -> None:
        if kv_dim <= 0:
            raise ValueError(f"kv_dim must be positive, got {kv_dim}")
        torch = _torch()
        self.kv_dim = int(kv_dim)
        # One projection per slot — keeps slot identities distinct
        # (emotion-slot weights cannot leak into music-slot output).
        self.emotion_proj = torch.nn.Linear(_EMOTION_DIM, self.kv_dim)
        self.music_proj = torch.nn.Linear(_MUSIC_DIM, self.kv_dim)
        self.time_proj = torch.nn.Linear(_TIME_DIM, self.kv_dim)
        self.provenance_proj = torch.nn.Linear(_PROVENANCE_DIM, self.kv_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, intent: Union[IntentFrame, Sequence[IntentFrame],
                                     Iterable[IntentFrame]]):
        return self.forward(intent)

    def forward(self, intent):
        torch = _torch()
        frames = self._coerce_to_list(intent)
        emo_feats = torch.tensor(
            [_featurize_emotion(f) for f in frames], dtype=torch.float32)
        mus_feats = torch.tensor(
            [_featurize_music(f) for f in frames], dtype=torch.float32)
        tim_feats = torch.tensor(
            [_featurize_time(f) for f in frames], dtype=torch.float32)
        prv_feats = torch.tensor(
            [_featurize_provenance(f) for f in frames], dtype=torch.float32)
        slots = [
            self.emotion_proj(emo_feats),
            self.music_proj(mus_feats),
            self.time_proj(tim_feats),
            self.provenance_proj(prv_feats),
        ]
        # Stack on the new "slot" axis at dim=1: each slot is (B, kv_dim).
        return torch.stack(slots, dim=1)

    @staticmethod
    def _coerce_to_list(intent) -> list[IntentFrame]:
        if isinstance(intent, IntentFrame):
            return [intent]
        out = list(intent)
        if not out:
            raise ValueError("intent iterable is empty")
        for i, frame in enumerate(out):
            if not isinstance(frame, IntentFrame):
                raise TypeError(
                    f"intent[{i}] must be an IntentFrame, got {type(frame).__name__}")
        return out

    # ------------------------------------------------------------------
    # Parameter convenience (matches CrossAttention's idiom)
    # ------------------------------------------------------------------

    def parameters(self):
        yield from self.emotion_proj.parameters()
        yield from self.music_proj.parameters()
        yield from self.time_proj.parameters()
        yield from self.provenance_proj.parameters()


__all__ = ["ConditioningProjection", "INTENT_SLOTS"]
