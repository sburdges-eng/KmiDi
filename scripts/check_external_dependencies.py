#!/usr/bin/env python3
"""
Pre-deletion dependency checker for KmiDi-compile.

Scans KmiDi-compile for hardcoded paths that point outside this folder.
Run before deleting sibling folders (AUDIO_MIDI_DATA, CANONICAL_REBUILD, etc.)
to see what would break.

Usage:
    python scripts/check_external_dependencies.py
    python scripts/check_external_dependencies.py --fix-env   # print .env.example additions
    python scripts/check_external_dependencies.py --json     # machine-readable report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_PATTERNS = [
    (r"/Users/seanburdges/[^\s\"'\\)]+", "absolute_home"),
    (r"/Volumes/[^\s\"'\\)]+", "absolute_volumes"),
    (r"~/Music[^\s\"'\\)]*", "home_music"),
    (r"~/Downloads[^\s\"'\\)]*", "home_downloads"),
    (r"~/Music-Brain[^\s\"'\\)]*", "home_music_brain"),
    (r"KELLY_AUDIO_DATA_ROOT\s*[=:]\s*[^\s\n]+", "env_kelly_audio"),
    (r"AUDIO_DATA_ROOT\s*[=:]\s*[^\s\n]+", "env_audio_data"),
    (r"RECOVERY_OPS|/Users/seanburdges/RECOVERY_OPS", "recovery_ops"),
    (
        r"AUDIO_MIDI_DATA|CANONICAL_REBUILD|FINAL_KMIDI|kelly-project|/KmiDi[^-]|sbdrive",
        "sibling_or_drive",
    ),
]


def scan_file(path: Path) -> List[Tuple[str, int, str, str]]:
    hits = []
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return hits
    for pattern, kind in EXTERNAL_PATTERNS:
        for m in re.finditer(pattern, text):
            line_num = text[: m.start()].count("\n") + 1
            snippet = m.group(0).strip()[:80]
            hits.append((kind, line_num, snippet, str(path.relative_to(ROOT))))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="Check for external path dependencies")
    ap.add_argument(
        "--fix-env", action="store_true", help="Print suggested env vars for .env.example"
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON report")
    args = ap.parse_args()

    kept_root = ROOT
    all_hits: List[dict] = []
    by_file: dict = {}
    by_kind: dict = {}

    for ext in ("py", "sh", "bash", "json", "yml", "yaml", "toml", "md", "env", "example"):
        for path in kept_root.rglob(f"*.{ext}"):
            if any(
                p in path.parts
                for p in (
                    ".git",
                    "node_modules",
                    "__pycache__",
                    "build",
                    ".venv",
                    "venv",
                    "external",
                )
            ):
                continue
            for kind, line, snippet, rel in scan_file(path):
                entry = {"kind": kind, "line": line, "snippet": snippet, "file": rel}
                all_hits.append(entry)
                by_file.setdefault(rel, []).append(entry)
                by_kind.setdefault(kind, []).append(entry)

    if args.json:
        print(
            json.dumps(
                {"by_file": by_file, "by_kind": by_kind, "total": len(all_hits)},
                indent=2,
            )
        )
        return 0

    if args.fix_env:
        print("# Add to KmiDi-compile/.env.example (or .env) before deleting sibling folders:\n")
        print("KELLY_AUDIO_DATA_ROOT=<KmiDi-compile>/datasets/audio")
        print("AUDIO_DATA_ROOT=<KmiDi-compile>/datasets/audio")
        print("# Optional: point to a local cache if you move AUDIO_MIDI_DATA into KmiDi-compile:")
        print("# KELLY_AUDIO_DATA_ROOT=<KmiDi-compile>/datasets/audio")
        return 0

    print("External path / dependency report (paths outside KmiDi-compile)\n")
    print("=" * 60)
    for rel in sorted(by_file.keys()):
        entries = by_file[rel]
        print(f"\n{rel}")
        for e in entries:
            print(f"  L{e['line']:4}  [{e['kind']}]  {e['snippet']}")
    print("\n" + "=" * 60)
    print(f"Total references: {len(all_hits)}")
    print("\nSafe to delete sibling folders if:")
    print("  - You set KELLY_AUDIO_DATA_ROOT (and optionally AUDIO_DATA_ROOT) to a path")
    print("    inside KmiDi-compile (e.g. KmiDi-compile/datasets/audio) or another drive.")
    print("  - You do not rely on RECOVERY_OPS, ~/Music, or /Volumes/sbdrive for runtime data.")
    print("  - Docs/vault paths (~/Music, ~/Downloads) are examples only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
