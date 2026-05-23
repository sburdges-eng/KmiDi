"""Tests for ``music_brain.latent.conditioning_bridge.ConditioningProjection``.

Closes the **Conditioning bridge projection layer** spec item. The
existing ``CrossAttention`` module (``music_brain/cross_attention.py``)
takes raw ``(B, S_kv, kv_dim)`` tensors and projects internally —
nothing in the tree turns an ``IntentFrame`` (the canonical
cross-language conditioning unit at
``shared_schemas/intent_frame_schema.json``) into that shape.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.cross_attention import CrossAttention  # noqa: E402
from music_brain.intent_ir import (  # noqa: E402
    EmotionState,
    IntentConstraints,
    IntentFrame,
    IntentMeta,
    IntentProvenance,
    IntentSource,
    MusicalIntent,
    TimeScope,
)
from music_brain.latent.conditioning_bridge import (  # noqa: E402
    INTENT_SLOTS,
    ConditioningProjection,
)


def _intent(*, valence: float = 0.5, arousal: float = 0.4) -> IntentFrame:
    return IntentFrame(
        meta=IntentMeta(intent_id=1, session_id=99),
        emotion=EmotionState(valence=valence, arousal=arousal, dominance=0.5,
                             confidence=0.9),
        music=MusicalIntent(tempo_bias=0.2, rhythmic_density=0.7,
                            groove_strength=0.6, harmonic_tension=0.3,
                            harmonic_motion=0.5, mode_preference=1,
                            melodic_activity=0.4, contour_variance=0.5,
                            dynamic_range=0.5, texture_density=0.5),
        time=TimeScope(start_bar=0, end_bar=8, fade_in_beats=0.0,
                       fade_out_beats=2.0),
        constraints=IntentConstraints(),
        provenance=IntentProvenance(source=IntentSource.UI_DIRECT,
                                    user_override_weight=1.0),
    )


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------

def test_construction_requires_positive_kv_dim() -> None:
    with pytest.raises(ValueError, match="kv_dim"):
        ConditioningProjection(kv_dim=0)


def test_slot_count_is_documented_constant() -> None:
    # Public stability: number of intent "tokens" the bridge emits.
    assert INTENT_SLOTS == 4


# ----------------------------------------------------------------------
# Shape contract
# ----------------------------------------------------------------------

def test_emits_kv_tensor_of_expected_shape() -> None:
    torch.manual_seed(0)
    bridge = ConditioningProjection(kv_dim=32)
    intent = _intent()
    kv = bridge(intent)
    assert kv.shape == (1, INTENT_SLOTS, 32)
    assert kv.dtype == torch.float32


def test_supports_batch_input() -> None:
    torch.manual_seed(0)
    bridge = ConditioningProjection(kv_dim=16)
    kv = bridge([_intent(valence=0.0), _intent(valence=-0.5),
                 _intent(valence=0.9)])
    assert kv.shape == (3, INTENT_SLOTS, 16)


# ----------------------------------------------------------------------
# Plugs into existing CrossAttention
# ----------------------------------------------------------------------

def test_output_is_consumable_by_cross_attention() -> None:
    torch.manual_seed(0)
    bridge = ConditioningProjection(kv_dim=32)
    attn = CrossAttention(query_dim=32, kv_dim=32, num_heads=4)

    intent = _intent()
    kv = bridge(intent)  # (1, INTENT_SLOTS, 32)
    query = torch.randn(1, 6, 32, dtype=torch.float32)  # 6 audio-latent steps

    out, weights = attn(query, kv)
    assert out.shape == (1, 6, 32)
    assert weights.shape == (1, 6, INTENT_SLOTS)
    # Attention weights are a valid distribution over kv slots.
    assert torch.allclose(weights.sum(dim=-1),
                          torch.ones(1, 6, dtype=torch.float32),
                          atol=1e-5)


# ----------------------------------------------------------------------
# Determinism / featurizer stability
# ----------------------------------------------------------------------

def test_same_intent_yields_same_kv() -> None:
    torch.manual_seed(0)
    bridge = ConditioningProjection(kv_dim=24)
    intent = _intent()
    kv_a = bridge(intent)
    kv_b = bridge(intent)
    assert torch.equal(kv_a, kv_b)


def test_different_intents_yield_different_kv() -> None:
    torch.manual_seed(0)
    bridge = ConditioningProjection(kv_dim=24)
    kv_a = bridge(_intent(valence=0.9, arousal=0.9))
    kv_b = bridge(_intent(valence=-0.9, arousal=-0.9))
    # Channels differ; not a coincidence on a tiny eps.
    assert not torch.allclose(kv_a, kv_b, atol=1e-3)


# ----------------------------------------------------------------------
# Featurizer is total over IntentFrame
# ----------------------------------------------------------------------

def test_featurizer_does_not_raise_on_default_intent() -> None:
    torch.manual_seed(0)
    bridge = ConditioningProjection(kv_dim=8)
    kv = bridge(IntentFrame())
    assert torch.all(torch.isfinite(kv))


def test_featurizer_handles_infinite_max_event_rate() -> None:
    """``IntentConstraints.max_event_rate`` defaults to +inf — must not
    propagate NaN/inf into the latent tensor."""
    torch.manual_seed(0)
    bridge = ConditioningProjection(kv_dim=8)
    intent = _intent()
    # default constraints already have max_event_rate=inf
    kv = bridge(intent)
    assert torch.all(torch.isfinite(kv))


def test_featurizer_is_parameter_owner() -> None:
    bridge = ConditioningProjection(kv_dim=8)
    params = list(bridge.parameters())
    assert params, "ConditioningProjection must own trainable parameters"
    for p in params:
        assert p.dtype == torch.float32
