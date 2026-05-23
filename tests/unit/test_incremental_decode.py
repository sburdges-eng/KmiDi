"""Tests for the incremental-decode loop (``music_brain.latent.decoding.incremental_decode``).

Closes **Incremental decoding pipeline** and **Streaming inference
support** (decoding side), composing the constraint sampler + KV-cache
+ scheduler into one realtime-aware loop.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.latent.decoding import DecodeConfig  # noqa: E402
from music_brain.latent.incremental_decode import (  # noqa: E402
    DecodeStep,
    incremental_decode,
)
from music_brain.latent.kv_cache import KVCache  # noqa: E402
from music_brain.latent.scheduler import JitterBoundedScheduler  # noqa: E402


def _toy_logit_fn(vocab_size: int):
    """A deterministic 'model': logit_i = a constant function of step
    and previous token — no real attention, but enough to test the loop."""

    def logit(step_idx: int, previous_token: int | None):
        torch.manual_seed(step_idx * 31 + (previous_token or 0))
        return torch.randn(1, vocab_size)

    return logit


def test_emits_max_steps_tokens_in_order() -> None:
    logit_fn = _toy_logit_fn(vocab_size=8)
    out = list(incremental_decode(
        logit_fn, max_steps=4,
        cfg=DecodeConfig(temperature=1e-6),  # greedy
    ))
    assert len(out) == 4
    assert all(isinstance(s, DecodeStep) for s in out)
    assert [s.step_index for s in out] == [0, 1, 2, 3]


def test_greedy_decode_is_deterministic() -> None:
    logit_fn = _toy_logit_fn(vocab_size=8)
    cfg = DecodeConfig(temperature=1e-6)
    a = [s.token_id for s in incremental_decode(logit_fn, max_steps=5, cfg=cfg)]
    b = [s.token_id for s in incremental_decode(logit_fn, max_steps=5, cfg=cfg)]
    assert a == b


def test_eos_token_terminates_loop_early() -> None:
    """Once the model emits ``eos_token``, the loop must stop."""
    vocab_size = 8

    def logit_fn(step_idx, previous_token):
        # Force token 7 to be sampled at step 2.
        logits = torch.full((1, vocab_size), -1e3)
        if step_idx == 2:
            logits[0, 7] = 100.0
        else:
            logits[0, step_idx] = 100.0
        return logits

    out = list(incremental_decode(
        logit_fn, max_steps=10, eos_token=7,
        cfg=DecodeConfig(temperature=1e-6),
    ))
    assert [s.token_id for s in out] == [0, 1, 7]


def test_kv_cache_grows_with_each_step() -> None:
    logit_fn = _toy_logit_fn(vocab_size=8)
    cache = KVCache(num_heads=1, head_dim=4, max_len=16)

    def attach(step_idx, previous_token):
        # Side-effect: append one slice per step to verify cache state.
        cache.append(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
        return logit_fn(step_idx, previous_token)

    list(incremental_decode(attach, max_steps=5,
                            cfg=DecodeConfig(temperature=1e-6)))
    assert cache.length == 5


def test_scheduler_records_step_latencies() -> None:
    logit_fn = _toy_logit_fn(vocab_size=8)
    sched = JitterBoundedScheduler(budget_ms=16.0)
    list(incremental_decode(logit_fn, max_steps=3,
                            cfg=DecodeConfig(temperature=1e-6),
                            scheduler=sched))
    assert sched.history_len == 3
    assert sched.mean_ms >= 0.0


def test_scheduler_under_pressure_forces_greedy_step() -> None:
    """When the scheduler reports thin headroom the decode must use
    greedy sampling — token must be argmax of the logits."""
    sched = JitterBoundedScheduler(budget_ms=16.0, greedy_threshold_ms=15.0)
    # Pre-fill history above threshold so headroom < greedy_threshold.
    for _ in range(4):
        sched.record_step_ms(15.0)

    # Logits with a *low* argmax — sampling would otherwise prefer the
    # large-logit token. We assert the *argmax* is what came out.
    def logit_fn(step_idx, previous_token):
        return torch.tensor([[10.0, -5.0, 0.0, 3.0]])

    cfg = DecodeConfig(temperature=2.0, top_k=4, top_p=1.0)
    out = list(incremental_decode(
        logit_fn, max_steps=1, cfg=cfg, scheduler=sched))
    # Greedy must pick the absolute argmax = token 0.
    assert out[0].token_id == 0
    assert out[0].mode == "greedy"


def test_forbidden_tokens_propagate() -> None:
    def logit_fn(step_idx, previous_token):
        # Argmax would normally be 0; with 0 forbidden, we get 3.
        return torch.tensor([[10.0, 1.0, 1.0, 5.0]])

    out = list(incremental_decode(
        logit_fn, max_steps=1,
        cfg=DecodeConfig(temperature=1e-6),
        forbidden_tokens=[0]))
    assert out[0].token_id == 3
