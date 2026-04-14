"""
Pydantic v1/v2 compatibility helpers.

Single source of truth for ``model → dict`` conversion across the codebase.
"""

from __future__ import annotations

from typing import Any, Dict


def pydantic_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a Pydantic model (v1 or v2) or dict-like object to a plain dict.

    Resolution order:
    1. ``model_dump()`` — Pydantic v2
    2. ``dict()`` — Pydantic v1
    3. ``obj`` itself if already a ``dict``
    4. ``vars(obj)`` as last resort

    Returns *obj* unchanged if it is already a dict.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return getattr(obj, "__dict__", {}) or {}
