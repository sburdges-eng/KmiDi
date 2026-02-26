"""
permission=GENERATE test.
Default revoked. Toggle grants/revokes. User-only.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui_gate.permission_generate import get_permission_generate, set_permission_generate


def test_default_revoked() -> None:
    """Default must be revoked."""
    set_permission_generate(False)  # Reset to default
    assert not get_permission_generate(), "Default must be revoked"


def test_grant_revoke() -> None:
    """Toggle grants and revokes."""
    set_permission_generate(False)
    assert not get_permission_generate()

    set_permission_generate(True)
    assert get_permission_generate()

    set_permission_generate(False)
    assert not get_permission_generate()


def main() -> int:
    try:
        test_default_revoked()
        test_grant_revoke()
        set_permission_generate(False)  # Restore default for other tests
        print("permission generate test: PASS")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
