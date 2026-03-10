"""
Shared UMP affect helpers: float-to-UMP32 and build MIDI 2.0 CC packets.

Used by scripts/ump_affect_harness.py and scripts/morph_affect_bridge.py.
Must match src/midi/AffectUMP.h and AffectUMP.cpp.
"""

# KmiDi affect vendor CC indices (must match src/midi/AffectUMP.h)
CC_VALENCE = 0x28
CC_AROUSAL = 0x29
CC_DYNAMICS = 0x2A

CONTROL_RATE_HZ = 100


def float_to_ump32(value: float, min_val: float, max_val: float) -> int:
    """Map float in [min_val, max_val] to 32-bit UMP [0, 0xFFFFFFFF]. Same as AffectUMP.cpp."""
    if min_val >= max_val:
        return 0
    clamped = max(min_val, min(max_val, value))
    norm = (clamped - min_val) / (max_val - min_val)
    v = int(round(norm * 0xFFFFFFFF))
    return min(v, 0xFFFFFFFF)


def build_ump_cc_v2(group: int, channel: int, controller: int, data32: int) -> tuple:
    """Build MIDI 2.0 Channel Voice CC (2 words) as 8 bytes, big-endian."""
    byte0 = (0x4 << 4) | (group & 0x0F)
    byte1 = (0x0B << 4) | (channel & 0x0F)
    byte2 = controller & 0x7F
    byte3 = 0
    word0 = (byte0 << 24) | (byte1 << 16) | (byte2 << 8) | byte3
    word1 = data32 & 0xFFFFFFFF
    return (
        (word0 >> 24) & 0xFF,
        (word0 >> 16) & 0xFF,
        (word0 >> 8) & 0xFF,
        word0 & 0xFF,
        (word1 >> 24) & 0xFF,
        (word1 >> 16) & 0xFF,
        (word1 >> 8) & 0xFF,
        word1 & 0xFF,
    )
