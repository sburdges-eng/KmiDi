"""
Phase tracking helpers used by the CLI.

Provides a minimal format_phase_progress helper to satisfy CLI/tests without
the full original implementation.
"""

from typing import List, Dict, Any


def format_phase_progress(phases: List[Dict[str, Any]] = None) -> str:
    """Return a simple string summary of phases."""
    if not phases:
        return "No phases tracked."
    lines = []
    for phase in phases:
        name = phase.get("name", "phase")
        status = phase.get("status", "unknown")
        progress = phase.get("progress", 0)
        lines.append(f"{name}: {status} ({progress}%)")
    return "\n".join(lines)


__all__ = ["format_phase_progress"]
