"""End-to-end test of the latent control core.

Composes every module touched by the latent-foundation PR into one
deterministic chain:

    synthetic mel
        → AudioJEPAEncoder.encode_to_frame
            → L2NormProjection
                → CrossAttention(query=audio_z, kv=ConditioningProjection(intent))
                    → WorldModel.rollout_frames
                        → stream_decode
                            → vector_store top-1 self-retrieval

This is the proof that the LatentFrame contract is sufficient to wire
the latent stack end-to-end. Determinism is asserted by seeding torch
and reconstructing the chain twice — every byte must match.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from music_brain.cross_attention import CrossAttention  # noqa: E402
from music_brain.intent_ir import (  # noqa: E402
    EmotionState,
    IntentFrame,
    IntentMeta,
    IntentProvenance,
    IntentSource,
    MusicalIntent,
    TimeScope,
)
from music_brain.jepa.audio_jepa import AudioJEPAEncoder  # noqa: E402
from music_brain.latent import (  # noqa: E402
    INTENT_SLOTS,
    ConditioningProjection,
    L2NormProjection,
    stream_decode,
)
from music_brain.vector_store import VectorStore  # noqa: E402
from music_brain.world_model import WorldModel  # noqa: E402


pytestmark = pytest.mark.integration


def _build_chain(seed: int = 0):
    """Construct the full latent pipeline with a deterministic seed."""
    torch.manual_seed(seed)
    encoder = AudioJEPAEncoder(tier="small", n_mels=32)  # latent_dim=128
    norm = L2NormProjection(radius=1.0)
    bridge = ConditioningProjection(kv_dim=encoder.latent_dim)
    attn = CrossAttention(query_dim=encoder.latent_dim, kv_dim=encoder.latent_dim, num_heads=4)
    world = WorldModel(state_dim=encoder.latent_dim, action_dim=0)
    return encoder, norm, bridge, attn, world


def _build_intent() -> IntentFrame:
    return IntentFrame(
        meta=IntentMeta(intent_id=42, session_id=7),
        emotion=EmotionState(valence=0.4, arousal=0.6, dominance=0.5, confidence=0.95),
        music=MusicalIntent(tempo_bias=0.1, rhythmic_density=0.7),
        time=TimeScope(start_bar=0, end_bar=8),
        provenance=IntentProvenance(source=IntentSource.UI_DIRECT, user_override_weight=1.0),
    )


def _run_chain(*, seed: int = 0, mel_seed: int = 1, horizon: int = 4):
    encoder, norm, bridge, attn, world = _build_chain(seed=seed)

    torch.manual_seed(mel_seed)
    # ``T`` (time) must be divisible by 4 because the JEPA encoder
    # downsamples freq-axis by 2^4 and uses stride 1 along time; we
    # leave time as-is. The audio frame returned by encode_to_frame
    # has shape (T, latent_dim).
    mel = torch.randn(1, 1, 32, 16, dtype=torch.float32)

    prov = IntentProvenance(source=IntentSource.UI_DIRECT, user_override_weight=1.0)
    initial = encoder.encode_to_frame(mel, time_index=0, provenance=prov, emotion_va=(0.4, 0.6))

    # Project encoder output back into the unit ball.
    norm_z = norm(initial.audio_z)
    bounded = initial.advance(time_index=initial.time_index + 1, audio_z=norm_z)  # time_index=1

    # Build conditioning context from the intent frame.
    kv = bridge(_build_intent())  # (1, INTENT_SLOTS, latent_dim)
    query = bounded.audio_z.unsqueeze(0)  # (1, T, latent_dim)
    conditioned, attn_weights = attn(query, kv)

    # Use the conditioned latents as the starting state for the world
    # model rollout (still wrapped in a LatentFrame so the contract
    # propagates).
    conditioned_frame = bounded.advance(
        time_index=bounded.time_index + 1, audio_z=conditioned.squeeze(0).contiguous()
    )  # time_index=2

    trajectory = world.rollout_frames(conditioned_frame, steps=horizon)

    return {
        "initial": initial,
        "bounded": bounded,
        "conditioned": conditioned_frame,
        "attn_weights": attn_weights,
        "trajectory": trajectory,
    }


# ----------------------------------------------------------------------
# Shape / contract assertions through the full chain
# ----------------------------------------------------------------------


def test_full_chain_shapes_align() -> None:
    result = _run_chain()
    latent_dim = result["initial"].audio_feature_dim
    assert result["bounded"].audio_feature_dim == latent_dim
    assert result["conditioned"].audio_feature_dim == latent_dim
    # Conditioning bridge produced INTENT_SLOTS kv tokens, so attention
    # weights sum-distribute over that slot dimension.
    assert result["attn_weights"].shape[-1] == INTENT_SLOTS
    assert torch.allclose(
        result["attn_weights"].sum(dim=-1),
        torch.ones_like(result["attn_weights"].sum(dim=-1)),
        atol=1e-5,
    )
    # Trajectory is 4 successive LatentFrames with monotonic time_index
    # starting at conditioned.time_index + 1.
    traj = result["trajectory"]
    assert len(traj) == 4
    base = result["conditioned"].time_index
    assert [f.time_index for f in traj] == [base + 1, base + 2, base + 3, base + 4]
    for f in traj:
        assert f.audio_z.shape == (1, latent_dim)
        assert f.audio_z.dtype == torch.float32


def test_trajectory_drains_through_stream_decode_in_order() -> None:
    result = _run_chain()
    traj = result["trajectory"]
    # Shuffle the producer order; stream_decode must restore monotonic order.
    permuted = [traj[2], traj[0], traj[3], traj[1]]
    out = list(stream_decode(permuted, start_index=traj[0].time_index))
    assert [f.time_index for f in out] == [f.time_index for f in traj]


def test_l2_projection_bounded_state_stays_bounded() -> None:
    result = _run_chain()
    # The bounded frame's audio_z sits inside the unit ball by
    # construction. We can't make the same claim of the conditioned
    # frame (attention is unbounded), but the upstream contract holds.
    norms = torch.linalg.norm(result["bounded"].audio_z, dim=-1)
    assert torch.all(norms <= 1.0 + 1e-5)


# ----------------------------------------------------------------------
# Vector-store retrieval round-trip
# ----------------------------------------------------------------------


def test_trajectory_self_retrieval_via_vector_store() -> None:
    result = _run_chain()
    traj = result["trajectory"]
    latent_dim = traj[0].audio_feature_dim
    store = VectorStore(dim=latent_dim)
    for frame in traj:
        # Flatten (1, D) → (D,) for the store.
        vec = frame.audio_z.squeeze(0).detach().cpu().numpy()
        store.add(f"t={frame.time_index}", vec, metadata={"t": frame.time_index})
    # Self-retrieve each frame — top-1 must be the same id.
    for frame in traj:
        q = frame.audio_z.squeeze(0).detach().cpu().numpy()
        hits = store.search(q, top_k=1)
        assert hits, "vector store returned no hits"
        top_id, top_score, _ = hits[0]
        assert top_id == f"t={frame.time_index}"
        assert top_score == pytest.approx(1.0, abs=1e-4)


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def test_full_chain_is_byte_identical_under_fixed_seed() -> None:
    a = _run_chain(seed=0, mel_seed=1)
    b = _run_chain(seed=0, mel_seed=1)
    for key in ("bounded", "conditioned"):
        assert torch.equal(
            a[key].audio_z, b[key].audio_z
        ), f"{key}.audio_z drifted under fixed seed"
    assert torch.equal(a["attn_weights"], b["attn_weights"])
    for fa, fb in zip(a["trajectory"], b["trajectory"]):
        assert torch.equal(fa.audio_z, fb.audio_z)
        assert fa.time_index == fb.time_index
    # And the dict round-trip is byte-stable too.
    a_dict = a["trajectory"][0].to_dict()
    b_dict = b["trajectory"][0].to_dict()
    assert np.array_equal(np.array(a_dict["audio_z"]), np.array(b_dict["audio_z"]))
