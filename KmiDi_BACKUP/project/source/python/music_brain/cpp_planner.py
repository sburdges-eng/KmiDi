"""
Stub C++ planner utilities used by the CLI.
"""

from typing import List, Dict, Any


def format_cpp_plan(plan: List[Dict[str, Any]] = None) -> str:
    if not plan:
        return "No C++ plan available."
    lines = []
    for step in plan:
        name = step.get("name", "step")
        status = step.get("status", "pending")
        lines.append(f"{name}: {status}")
    return "\n".join(lines)


__all__ = ["format_cpp_plan"]
