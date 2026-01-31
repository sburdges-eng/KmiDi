#!/usr/bin/env python3
"""
Phase 4 stub-creep check (reports undocumented stub/forensic phrases).

Searches active code for undocumented stubs and "restore from forensic if needed".
Exits 0 if no matches (or only in allowed paths); exits 1 with report otherwise.

Usage:
  python scripts/check_stub_creep.py [--allow-docs] [ROOT]
  ROOT defaults to repo root (parent of scripts/).

Allowed: docstrings and comments that document deferral (e.g. "status: stubbed" in contracts).
Excluded: __pycache__, .git, *.pyc, docs/*.md, BUILD, dist, .venv, node_modules.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Patterns that indicate stub creep (undocumented stub or forensic fallback)
STUB_PATTERN = re.compile(
    r"\b(stub|stubbed|stubs)\b",
    re.IGNORECASE,
)
FORENSIC_PATTERN = re.compile(
    r"restore\s+from\s+forensic\s+if\s+(needed|required)",
    re.IGNORECASE,
)

# Lines that are explicit documentation (allowed)
ALLOWED_CONTEXTS = (
    "status",
    "stubbed",
    "completed",
    "failed",
    "CONTRACTS",
    "BOOT",
    "details",
    "docstring",
    "documented",
    "deferral",
    "deferred",
    "placeholder",
    "optional",
    "when not installed",
    "when unavailable",
    "if needed",
    "per §",
    "per docs",
    "see docs",
    "see CONTRACTS",
    "returns dict with status",
    "status in {",
    "status:",
    '"stubbed"',
    "'stubbed'",
    "orchestrator",
    "Tauri",
    "without Logic",
    "Phase 4",
    "validation",
    "echo response",
    "otherwise stub",
)


def _is_allowed_line(line: str, path: Path) -> bool:
    """Allow docstrings, comments, and known contract text."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    # Allow if line is clearly documenting the contract
    lower = line.lower()
    for ctx in ALLOWED_CONTEXTS:
        if ctx.lower() in lower:
            return True
    # Allow in docs
    if path.suffix == ".md" or "docs" in path.parts:
        return True
    return False


def scan_file(path: Path, allow_docs: bool) -> list[tuple[int, str, str]]:
    """Return list of (line_no, line, match_type) for matches."""
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return hits
    for i, line in enumerate(text.splitlines(), 1):
        if allow_docs and _is_allowed_line(line, path):
            continue
        if FORENSIC_PATTERN.search(line):
            hits.append((i, line.strip(), "forensic"))
        elif STUB_PATTERN.search(line) and not _is_allowed_line(line, path):
            hits.append((i, line.strip(), "stub"))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for undocumented stub/forensic phrases; exit 1 if found."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Repo root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--allow-docs",
        action="store_true",
        help="Allow matches in docstrings/comments that document deferral",
    )
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).resolve()
    else:
        root = Path(__file__).resolve().parent.parent
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    exclude_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build"}
    exclude_suffixes = {".pyc", ".pyo", ".egg-info"}
    # Only scan canonical repo (KmiDi MIDI Companion); skip Desktop, backups, other trees
    exclude_path_contains = ("Desktop", "SSD_Transfer", "final kel", "KmiDi not fixed", "KmiDi-remote", "KmiDi v.", "kelly-music-brain-backup", "kelly-music-brain-clean")

    all_hits: list[tuple[Path, int, str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(p in rel.parts for p in exclude_dirs):
            continue
        if any(seg in str(rel) for seg in exclude_path_contains):
            continue
        if path.suffix in exclude_suffixes:
            continue
        if path.suffix != ".py":
            continue
        if path.name == "check_stub_creep.py":
            continue
        hits = scan_file(path, args.allow_docs)
        for line_no, line, match_type in hits:
            all_hits.append((path, line_no, line, match_type))

    if not all_hits:
        print("check_stub_creep: no undocumented stub/forensic matches.")
        return 0

    print("check_stub_creep: found matches (fix or convert to documented deferral):", file=sys.stderr)
    for path, line_no, line, match_type in all_hits:
        short = path.relative_to(root) if root in path.parents else path
        print(f"  {short}:{line_no} [{match_type}] {line[:80]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
