"""
Probe tone guardrail: PROBE_TONE_* must be fixed constants.
They must NEVER be derived from intent, value states, or harmony.
If someone ties pitch to harmony here, CI should scream.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui_gate.audio_playback import PROBE_TONE_A, PROBE_TONE_B, PROBE_TONE_C


def test_probe_tones_are_fixed() -> int:
    """Probe tones must be fixed. Not derived from intent or value states."""
    assert PROBE_TONE_A == 440.0
    assert PROBE_TONE_B == 554.37
    assert PROBE_TONE_C == 659.25
    return 0


def main() -> int:
    try:
        test_probe_tones_are_fixed()
        print("probe tone guardrail: PASS")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
