"""
Programmable Button — permission=GENERATE.

Optional, scalable. User must explicitly grant permission to generate.
Never inferred from intent, value states, or MIDI.
"""

from __future__ import annotations

_permission_generate: bool = False  # Default: revoked


def get_permission_generate() -> bool:
    """Returns whether user has granted generate permission."""
    return _permission_generate


def set_permission_generate(granted: bool) -> None:
    """Set generate permission. User-only; called from button click."""
    global _permission_generate
    _permission_generate = granted
