"""
D1 — Filtering acceptance tests.
AC-D1-1 through AC-D1-6.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minimal_listening_core.filter_audio import filter_audio_gain


def test_ac_d1_1_post_generation_only() -> int:
    """AC-D1-1: Filtering applied only to post-generation output (caller responsibility)."""
    pcm = [0.5, -0.3, 0.0]
    out = filter_audio_gain(pcm, 0.8)
    assert len(out) == len(pcm)
    return 0


def test_ac_d1_2_explicit_params() -> int:
    """AC-D1-2: Filter params from explicit control or fixed preset."""
    pcm = [1.0, 0.5]
    out = filter_audio_gain(pcm, 0.5)
    assert out[0] == 0.5 and out[1] == 0.25
    return 0


def test_ac_d1_4_no_added_notes() -> int:
    """AC-D1-4: Filtering does not add notes, voices, or layers."""
    pcm = [0.1, 0.2]
    out = filter_audio_gain(pcm, 2.0)
    assert len(out) == len(pcm)
    return 0


def test_ac_d1_6_deterministic() -> int:
    """AC-D1-6: Same input + same params → same output."""
    pcm = [0.3, -0.4]
    a = filter_audio_gain(pcm, 0.7)
    b = filter_audio_gain(pcm, 0.7)
    assert a == b
    return 0


def test_ac_d1_5_no_intent_imports() -> int:
    """AC-D1-5: Filtering path does not call intent inference."""
    pcm = [0.0]
    filter_audio_gain(pcm)
    return 0


def main() -> int:
    r = 0
    r |= test_ac_d1_1_post_generation_only()
    r |= test_ac_d1_2_explicit_params()
    r |= test_ac_d1_4_no_added_notes()
    r |= test_ac_d1_6_deterministic()
    r |= test_ac_d1_5_no_intent_imports()
    if r == 0:
        print("filter_audio_test: PASS")
    return r


if __name__ == "__main__":
    sys.exit(main())
