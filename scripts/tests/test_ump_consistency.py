"""
Cross-language consistency tests for float → UMP32 mapping.

Python (scripts/ump_affect_utils.py) and C++ (src/midi/AffectUMP.cpp) must
produce the same outputs for the same inputs. These canonical values define
the contract.
"""
import sys
from pathlib import Path

# Allow importing from scripts/ when run as pytest scripts/tests/ or standalone
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from ump_affect_utils import float_to_ump32

# Canonical expected results (aligned with C++ floatToUmp32Value in AffectUMP.cpp)
EXPECTED_RESULTS = {
    (-1.0, -1.0, 1.0): 0,
    (0.0, -1.0, 1.0): 0x80000000,
    (1.0, -1.0, 1.0): 0xFFFFFFFF,
    (0.5, 0.0, 1.0): 0x80000000,
}


def test_float_to_ump32_canonical():
    """Canonical float→UMP32 mappings must match Python and C++ contract."""
    for (value, min_val, max_val), expected in EXPECTED_RESULTS.items():
        py_val = float_to_ump32(value, min_val, max_val)
        assert py_val == expected, (
            f"float_to_ump32({value}, {min_val}, {max_val}) = {py_val} != {expected}"
        )
