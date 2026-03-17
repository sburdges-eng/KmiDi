# MIDI and UMP (MIDI 2.0) utilities

This directory contains C++ code for MIDI 2.0 affect mapping and generation. The float→UMP32 mapping must stay in sync with the Python implementation used by scripts and the API.

## Float → UMP32 mapping (canonical)

The function `floatToUmp32Value` in [AffectUMP.cpp](AffectUMP.cpp) maps a float in `[minVal, maxVal]` to a 32-bit UMP value in `[0, 0xFFFFFFFF]`. Python equivalent: [scripts/ump_affect_utils.py](../../scripts/ump_affect_utils.py) `float_to_ump32`.

**Canonical mappings (Python and C++ must match):**

| value | minVal | maxVal | UMP32 (hex) | UMP32 (decimal) |
|-------|--------|--------|-------------|-----------------|
| -1.0  | -1.0   | 1.0    | 0x0         | 0               |
| 0.0   | -1.0   | 1.0    | 0x80000000  | 2147483648      |
| 1.0   | -1.0   | 1.0    | 0xFFFFFFFF  | 4294967295      |
| 0.5   | 0.0    | 1.0    | 0x80000000  | 2147483648      |

Formula: clamp value to [minVal, maxVal], then `norm = (clamped - minVal) / (maxVal - minVal)`, then `round(norm * 0xFFFFFFFF)` capped at 0xFFFFFFFF.

Cross-language tests: [scripts/tests/test_ump_consistency.py](../../scripts/tests/test_ump_consistency.py).
