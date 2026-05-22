"""Tests for ``music_brain.lora_adapter.LoRAAdapter``.

Skips cleanly when torch is unavailable in the active pytest environment.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from music_brain.lora_adapter import LoRAAdapter  # noqa: E402


def _seeded() -> None:
    torch.manual_seed(0)


def _linear(in_f: int = 8, out_f: int = 4) -> torch.nn.Linear:
    _seeded()
    return torch.nn.Linear(in_f, out_f, bias=True)


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------

def test_rank_must_be_non_negative() -> None:
    base = _linear()
    with pytest.raises(ValueError, match="rank"):
        LoRAAdapter(base, rank=-1)


def test_rank_zero_acts_as_identity_adapter() -> None:
    base = _linear()
    adapter = LoRAAdapter(base, rank=0)
    x = torch.randn(3, 8)
    y_base = base(x)
    y_adapter = adapter(x)
    assert torch.allclose(y_base, y_adapter), "rank=0 must equal base forward"
    assert adapter.num_trainable_parameters() == 0


def test_in_out_features_override() -> None:
    """Bases without .in_features/.out_features can be wrapped explicitly."""
    base = torch.nn.Sequential(torch.nn.Linear(8, 4))  # no in_features attr
    adapter = LoRAAdapter(base, rank=2, in_features=8, out_features=4)
    assert adapter.in_features == 8
    assert adapter.out_features == 4


# ----------------------------------------------------------------------
# Forward correctness
# ----------------------------------------------------------------------

def test_b_initialised_to_zero_so_initial_forward_equals_base() -> None:
    base = _linear()
    adapter = LoRAAdapter(base, rank=4)
    x = torch.randn(2, 8)
    assert torch.allclose(adapter(x), base(x)), (
        "Standard LoRA init has B=0 so the initial output equals base"
    )


def test_forward_changes_after_nonzero_b() -> None:
    base = _linear()
    adapter = LoRAAdapter(base, rank=4, alpha=4.0)  # scaling = 1.0
    x = torch.randn(2, 8)
    # Replace B with nonzero so the adapter takes effect.
    with torch.no_grad():
        adapter.B.fill_(0.1)
    y = adapter(x)
    expected = base(x) + 1.0 * (x @ adapter.A.t() @ adapter.B.t())
    assert torch.allclose(y, expected, atol=1e-6)


def test_shape_preserved_across_extra_batch_dims() -> None:
    base = _linear(in_f=6, out_f=3)
    adapter = LoRAAdapter(base, rank=2)
    with torch.no_grad():
        adapter.B.fill_(0.01)
    x = torch.randn(5, 7, 6)  # (..., in_features)
    y = adapter(x)
    assert y.shape == (5, 7, 3)


def test_alpha_defaults_to_rank_for_scaling_one() -> None:
    base = _linear()
    adapter = LoRAAdapter(base, rank=4)  # alpha defaults to rank
    assert adapter.scaling == pytest.approx(1.0)


def test_alpha_can_be_overridden() -> None:
    base = _linear()
    adapter = LoRAAdapter(base, rank=4, alpha=16.0)
    assert adapter.scaling == pytest.approx(4.0)


# ----------------------------------------------------------------------
# Training ergonomics
# ----------------------------------------------------------------------

def test_freeze_base_disables_grad_on_base_only() -> None:
    base = _linear()
    adapter = LoRAAdapter(base, rank=4)
    adapter.freeze_base()
    for p in base.parameters():
        assert p.requires_grad is False
    for p in adapter.trainable_parameters():
        assert p.requires_grad is True


def test_trainable_parameter_count() -> None:
    base = _linear(in_f=8, out_f=4)
    adapter = LoRAAdapter(base, rank=2)
    # A is (rank, in) = 2x8 = 16; B is (out, rank) = 4x2 = 8; total 24.
    assert adapter.num_trainable_parameters() == 24


def test_freeze_is_idempotent() -> None:
    base = _linear()
    adapter = LoRAAdapter(base, rank=4)
    adapter.freeze_base()
    adapter.freeze_base()  # must not raise
    assert all(not p.requires_grad for p in base.parameters())


# ----------------------------------------------------------------------
# Backprop sanity
# ----------------------------------------------------------------------

def test_gradients_flow_to_adapter_but_not_to_frozen_base() -> None:
    base = _linear()
    adapter = LoRAAdapter(base, rank=4)
    adapter.freeze_base()
    with torch.no_grad():
        adapter.B.fill_(0.1)
    x = torch.randn(3, 8)
    target = torch.randn(3, 4)
    y = adapter(x)
    loss = (y - target).pow(2).mean()
    loss.backward()
    # Base layer params: gradient should be None (no autograd attached).
    for p in base.parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            "frozen base must not accumulate gradient"
        )
    # Adapter params: gradient must be non-zero somewhere.
    assert adapter.A.grad is not None
    assert adapter.B.grad is not None
    assert adapter.A.grad.abs().sum().item() > 0
    assert adapter.B.grad.abs().sum().item() > 0
