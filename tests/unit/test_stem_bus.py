"""StemBus — multi-stem audio container with mix-down and per-stem gain.

Goal: Stem-aware generation, Multi-instrument orchestration."""

import numpy as np
import pytest

from music_brain.audio.stems import StemBus

# ---------------------------------------------------------------------------
# Construction and assignment
# ---------------------------------------------------------------------------


def test_bus_starts_empty():
    bus = StemBus(num_samples=64)
    assert bus.names == []
    assert bus.mix().tolist() == [0.0] * 64


def test_set_stem_stores_buffer():
    bus = StemBus(num_samples=4)
    bus.set("drums", np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    assert bus.names == ["drums"]
    np.testing.assert_array_equal(bus.get("drums"), [1.0, 1.0, 1.0, 1.0])


def test_set_stem_with_wrong_length_raises():
    bus = StemBus(num_samples=4)
    with pytest.raises(ValueError, match="length"):
        bus.set("drums", np.array([1.0, 1.0], dtype=np.float32))


def test_get_unknown_stem_raises():
    bus = StemBus(num_samples=4)
    with pytest.raises(KeyError):
        bus.get("missing")


# ---------------------------------------------------------------------------
# mix() with per-stem gain
# ---------------------------------------------------------------------------


def test_mix_sums_stems():
    bus = StemBus(num_samples=4)
    bus.set("a", np.ones(4, dtype=np.float32))
    bus.set("b", np.ones(4, dtype=np.float32) * 2.0)
    assert bus.mix().tolist() == [3.0, 3.0, 3.0, 3.0]


def test_mix_applies_per_stem_gain():
    bus = StemBus(num_samples=4)
    bus.set("a", np.ones(4, dtype=np.float32))
    bus.set("b", np.ones(4, dtype=np.float32))
    bus.set_gain("a", 0.5)
    bus.set_gain("b", 2.0)
    assert bus.mix().tolist() == [2.5, 2.5, 2.5, 2.5]


def test_set_gain_unknown_stem_raises():
    bus = StemBus(num_samples=4)
    with pytest.raises(KeyError):
        bus.set_gain("missing", 1.0)


def test_default_gain_is_one():
    bus = StemBus(num_samples=4)
    bus.set("a", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert bus.get_gain("a") == 1.0
    np.testing.assert_array_equal(bus.mix(), [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# mute / solo
# ---------------------------------------------------------------------------


def test_mute_excludes_stem_from_mix():
    bus = StemBus(num_samples=4)
    bus.set("a", np.ones(4, dtype=np.float32))
    bus.set("b", np.ones(4, dtype=np.float32))
    bus.mute("a", True)
    assert bus.mix().tolist() == [1.0, 1.0, 1.0, 1.0]


def test_unmute_restores_stem():
    bus = StemBus(num_samples=4)
    bus.set("a", np.ones(4, dtype=np.float32))
    bus.mute("a", True)
    bus.mute("a", False)
    assert bus.mix().tolist() == [1.0, 1.0, 1.0, 1.0]


def test_solo_includes_only_soloed_stems():
    bus = StemBus(num_samples=4)
    bus.set("a", np.ones(4, dtype=np.float32))
    bus.set("b", np.ones(4, dtype=np.float32) * 2.0)
    bus.set("c", np.ones(4, dtype=np.float32) * 4.0)
    bus.solo("b", True)
    # Only b is audible.
    assert bus.mix().tolist() == [2.0, 2.0, 2.0, 2.0]


def test_solo_overrides_mute():
    bus = StemBus(num_samples=4)
    bus.set("a", np.ones(4, dtype=np.float32))
    bus.set("b", np.ones(4, dtype=np.float32) * 3.0)
    bus.mute("b", True)
    bus.solo("b", True)
    # solo wins over mute (DAW convention).
    assert bus.mix().tolist() == [3.0, 3.0, 3.0, 3.0]


# ---------------------------------------------------------------------------
# Iteration / inspection
# ---------------------------------------------------------------------------


def test_names_preserves_insertion_order():
    bus = StemBus(num_samples=4)
    bus.set("b", np.ones(4, dtype=np.float32))
    bus.set("a", np.ones(4, dtype=np.float32))
    bus.set("c", np.ones(4, dtype=np.float32))
    assert bus.names == ["b", "a", "c"]


def test_remove_stem_drops_from_mix():
    bus = StemBus(num_samples=4)
    bus.set("a", np.ones(4, dtype=np.float32))
    bus.set("b", np.ones(4, dtype=np.float32) * 2.0)
    bus.remove("a")
    assert bus.names == ["b"]
    assert bus.mix().tolist() == [2.0, 2.0, 2.0, 2.0]
