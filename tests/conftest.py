"""Pytest conftest: add repo root and brain to path so music_brain and mcp_workstation resolve."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRAIN = ROOT / "KmiDi_CANON" / "brain"
for p in (ROOT, BRAIN):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
