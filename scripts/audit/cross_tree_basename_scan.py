#!/usr/bin/env python3
"""
Cross-tree basename collision scanner.

Detects `.cpp` files that share a basename across `src/` and `src_penta-core/`.
When both trees compile into the same library graph (KellyCore links
penta_core PUBLIC), identical basenames with overlapping namespace/class
definitions cause ODR violations — the linker silently picks one, results
depend on link order, and bugs cascade unpredictably.

Caught the 2026-Q2 RTLogger / DiagnosticsEngine / PerformanceMonitor /
AudioAnalyzer patterns during the audit. Intended as a pre-build CI gate
so new branches can't silently reintroduce the same class of bug.

Usage:
    scripts/audit/cross_tree_basename_scan.py
    scripts/audit/cross_tree_basename_scan.py --json

Exits 0 when no cross-tree basename collision exists; 1 otherwise.

Complements:
  - `scripts/audit/intra_lib_odr_scan.py` (post-build symbol collision)
  - `scripts/audit/odr_pair_diff.sh` (per-file API diff for a hardcoded
    pair list; this scanner discovers pairs dynamically)
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

TREES = [REPO / "src", REPO / "src_penta-core"]

# Allowed overlap: explicit namespace separation documented per file.
# Add entries with a one-line justification. Each value is the reason
# both copies can coexist without ODR conflict.
ALLOWLIST: dict[str, str] = {
    # Same basename but one is midikompanion::audio::AudioAnalyzer, the
    # other is penta::diagnostics::AudioAnalyzer — distinct classes.
    # The duplicate-in-`penta::diagnostics` was deleted in PR #156.
    # "AudioAnalyzer.cpp": "midikompanion::audio vs penta::diagnostics",
    # Same basename but different kelly-namespace classes:
    #   src/midi/GrooveEngine.cpp       → kelly::GrooveEngine
    #   src/engines/GrooveEngine.cpp    → kelly::GroovePatternEngine (renamed)
    #   src_penta-core/groove/...       → penta::groove::GrooveEngine
    "GrooveEngine.cpp": "kelly::GrooveEngine vs kelly::GroovePatternEngine vs penta::groove::GrooveEngine",
}

NAMESPACE_RE = re.compile(r"^namespace\s+([\w:]+)(?:\s*::\s*(\w+))?\s*\{?", re.MULTILINE)


def extract_top_namespace(cpp: Path) -> str:
    """Return the top-level namespace as a string ('penta::diagnostics', 'kelly', etc.)."""
    try:
        head = cpp.read_text(errors="ignore")[:2048]
    except Exception:
        return ""
    match = NAMESPACE_RE.search(head)
    if not match:
        return ""
    first, second = match.group(1), match.group(2)
    return f"{first}::{second}" if second else first


def scan() -> dict[str, list[tuple[str, str]]]:
    """Return basename → list of (relpath, namespace) across all trees."""
    table: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tree in TREES:
        if not tree.exists():
            continue
        for cpp in tree.rglob("*.cpp"):
            rel = str(cpp.relative_to(REPO))
            ns = extract_top_namespace(cpp)
            table[cpp.name].append((rel, ns))
    return {k: v for k, v in table.items() if len(v) >= 2}


def main(argv: list[str]) -> int:
    collisions = scan()
    real: list[dict[str, object]] = []

    for basename, entries in collisions.items():
        # Split by top-level namespace; same-namespace duplicates are ODR bugs.
        by_ns: dict[str, list[str]] = defaultdict(list)
        for rel, ns in entries:
            by_ns[ns].append(rel)
        dupe_in_ns = {ns: paths for ns, paths in by_ns.items() if len(paths) >= 2}
        if not dupe_in_ns:
            continue
        if basename in ALLOWLIST:
            continue
        real.append({
            "basename": basename,
            "duplicates_by_namespace": dupe_in_ns,
        })

    if "--json" in argv:
        json.dump(real, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if real:
            print(f"Found {len(real)} cross-tree basename collision(s) in overlapping namespaces:")
            print()
            for item in real:
                print(f"  {item['basename']}:")
                for ns, paths in item["duplicates_by_namespace"].items():
                    label = ns or "(global)"
                    print(f"    namespace {label}")
                    for p in paths:
                        print(f"      - {p}")
            print()
            print("These files share a basename AND a top-level namespace — linker will")
            print("see duplicate symbols.  Either (a) delete the stale copy, (b) rename")
            print("one of them, (c) if they're intentionally distinct classes, add the")
            print("basename to ALLOWLIST in this script with a justification.")
        else:
            print("No cross-tree basename/namespace collisions.")

    return 1 if real else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
