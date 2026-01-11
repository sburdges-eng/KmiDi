"""
Ensure local repo code is imported first during test runs.

Python automatically imports `sitecustomize` if present on sys.path. We place
this in the repo root to:
- prepend the repo root to sys.path
- drop conflicting paths from other checkouts (e.g., miDiKompanion)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_PY = ROOT / "source" / "python"


def _prepend(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


# Prefer consolidated source layout first, then repo root, then drop conflicts.
_prepend(SOURCE_PY)
_prepend(ROOT)


def _is_conflict(path: str) -> bool:
    return "miDiKompanion" in path


sys.path = [p for p in sys.path if not _is_conflict(p)]

