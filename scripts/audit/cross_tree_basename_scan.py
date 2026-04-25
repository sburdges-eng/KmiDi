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

# Allowed overlap: explicit (basename, namespace) pair documented per file.
# Keying on the *pair* (not just the basename) prevents a future regression
# where someone adds a third copy in a different namespace under an existing
# allowlisted basename — that case must still be flagged.
#
# To suppress a known-good (basename, ns) collision, add an entry with a
# one-line justification.
ALLOWLIST: dict[tuple[str, str], str] = {
    # Distinct classes that share a basename inside the kelly namespace:
    #   src/midi/GrooveEngine.cpp       → class kelly::GrooveEngine
    #   src/engines/GrooveEngine.cpp    → class kelly::GroovePatternEngine (renamed)
    # Both live in `namespace kelly`, but expose distinct symbols.
    # A third copy under (`GrooveEngine.cpp`, `penta::groove`) is intentional
    # and lives in a different namespace, so the scanner reports it as 1 path,
    # not a collision.
    ("GrooveEngine.cpp", "kelly"):
        "kelly::GrooveEngine vs kelly::GroovePatternEngine — distinct classes",
}

# Match a `namespace <X>{::<Y>}* {` opener, optionally with the C++17
# nested-name `namespace foo::bar` form.  Anchoring at column 0 keeps us
# from matching `using namespace ...;` inside function bodies.
NS_OPEN_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][\w:]*)\s*\{", re.MULTILINE)


def extract_top_namespace(cpp: Path) -> str:
    """Return the file's top-level namespace path as a `::`-joined string.

    Handles both forms equivalently so they collide as expected:
        namespace penta::diagnostics { ... }
        namespace penta { namespace diagnostics { ... } }
    """
    try:
        text = cpp.read_text(errors="ignore")[:4096]
    except Exception:
        return ""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)

    first = NS_OPEN_RE.search(text)
    if not first:
        return ""
    components = [first.group(1)]
    rest = text[first.end():]

    while True:
        m = re.match(r"\s*namespace\s+([A-Za-z_][\w:]*)\s*\{", rest)
        if not m:
            break
        components.append(m.group(1))
        rest = rest[m.end():]

    return "::".join(components)


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
        by_ns: dict[str, list[str]] = defaultdict(list)
        for rel, ns in entries:
            by_ns[ns].append(rel)
        dupe_in_ns = {
            ns: paths for ns, paths in by_ns.items() if len(paths) >= 2
        }
        if not dupe_in_ns:
            continue
        unallowed = {
            ns: paths
            for ns, paths in dupe_in_ns.items()
            if (basename, ns) not in ALLOWLIST
        }
        if not unallowed:
            continue
        real.append({
            "basename": basename,
            "duplicates_by_namespace": unallowed,
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
