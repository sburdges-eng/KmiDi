#!/usr/bin/env python3
"""
Intra-library ODR scanner.

Catches symbols defined in two or more `.cpp.o` files within the SAME
static-library build directory — the duplicate-symbol class that the
existing per-pair script (odr_pair_diff.sh) misses because it only
compares src/ vs src_penta-core/. The bug pattern that motivated this
scanner was found 2026-04-14: ChordAnalyzer.cpp and ChordAnalyzerSIMD.cpp
both defined `analyzeSIMD` via different `#ifdef __AVX2__` branches that
both happened to compile on Apple Silicon. Linker silently picked one;
non-deterministic which.

Usage:
    scripts/audit/intra_lib_odr_scan.py

Exits non-zero if any duplicate-definition is found. Intended as a CI
gate after `cmake --build build`.

Heuristics:
  - Only `T` symbols (defined text/code) are flagged. `t` (local) and
    `S`/`s` (debug/section) are ignored. `r`/`R` (read-only data) is
    expected to dedup via inline-vague-linkage and is not flagged.
  - Symbols starting with `__Z` are demangled via c++filt for the report.
  - Inline functions, templates, and vague-linkage symbols (the bulk of
    expected duplicates) are filtered: they appear with `S _foo` in nm
    output (weak/coalesced), not `T _foo`.
"""
from __future__ import annotations

import collections
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build"


def find_libs() -> dict[str, list[Path]]:
    """Group .cpp.o files by the static-lib directory that owns them."""
    libs: dict[str, list[Path]] = collections.defaultdict(list)
    for obj in BUILD.rglob("*.cpp.o"):
        # Library = first directory under CMakeFiles, e.g.
        # build/src_penta-core/CMakeFiles/penta_core.dir/...
        parts = obj.relative_to(BUILD).parts
        if "CMakeFiles" not in parts:
            continue
        idx = parts.index("CMakeFiles")
        # The lib name is the directory immediately after CMakeFiles,
        # stripped of '.dir' suffix.
        if idx + 1 >= len(parts):
            continue
        lib_dir = parts[idx + 1]
        if not lib_dir.endswith(".dir"):
            continue
        lib_name = lib_dir[:-4]  # 'penta_core.dir' -> 'penta_core'
        libs[lib_name].append(obj)
    return libs


def text_symbols(obj: Path) -> set[str]:
    """STRONG (non-weak) external text symbols defined in this object.

    Uses `nm -m` instead of `nm -U` because Mach-O distinguishes weak
    vague-linkage symbols (template instantiations, inline functions —
    the linker is supposed to dedup these) from strong globals only via
    the `-m` linkage column. Without this filter the scanner emits
    thousands of false positives from libc++ template inliners.

    `nm -m` output format:
        <address> (<section>) <linkage> <name>
    where <linkage> is e.g. "external", "weak external", "weak private
    external", "non-external", etc. We keep ONLY symbols whose linkage
    is exactly "external" — those are the real ODR violations.
    """
    try:
        out = subprocess.run(
            ["nm", "-m", str(obj)],
            check=True, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError:
        return set()
    syms: set[str] = set()
    for line in out.splitlines():
        # Skip undefined references (no address)
        if line.startswith(" "):
            continue
        # Strip address + section
        rest = line.split(") ", 1)
        if len(rest) != 2:
            continue
        # Now rest[1] is "<linkage> <name>"
        # We want sections that are __TEXT (skip __DATA, __BSS, __OBJC, etc.)
        if "__TEXT" not in rest[0]:
            continue
        tail = rest[1].strip()
        # Linkage is the longest prefix of words that aren't the symbol;
        # the symbol itself starts with `_` (Mach-O leading underscore).
        underscore = tail.find(" _")
        if underscore < 0:
            continue
        linkage = tail[:underscore].strip()
        name = tail[underscore + 1:].strip()
        # Strong global only
        if linkage != "external":
            continue
        if name.startswith(("ltmp", "Lfunc_begin")):
            continue
        syms.add(name)
    return syms


def demangle(symbols: list[str]) -> dict[str, str]:
    """Best-effort C++ demangle via c++filt over stdin (avoids argv length limits)."""
    if not symbols or not shutil.which("c++filt"):
        return {s: s for s in symbols}
    stdin = "\n".join(symbols)
    try:
        out = subprocess.run(
            ["c++filt"], input=stdin, check=True,
            capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError:
        return {s: s for s in symbols}
    lines = out.strip().splitlines()
    if len(lines) != len(symbols):
        return {s: s for s in symbols}
    return dict(zip(symbols, lines))


def main() -> int:
    if not BUILD.is_dir():
        print(f"build/ not found at {BUILD} — run cmake --build first")
        return 2

    libs = find_libs()
    if not libs:
        print("no .cpp.o files found under build/")
        return 2

    total_dupes = 0
    print("# Intra-Library ODR Scan")
    print()
    print(f"Scanned {sum(len(v) for v in libs.values())} object files "
          f"across {len(libs)} static libraries.")
    print()

    for lib in sorted(libs):
        objs = libs[lib]
        # Map symbol → list of object files defining it
        sym_to_objs: dict[str, list[Path]] = collections.defaultdict(list)
        for obj in objs:
            for sym in text_symbols(obj):
                sym_to_objs[sym].append(obj)

        dupes = {s: v for s, v in sym_to_objs.items() if len(v) > 1}
        if not dupes:
            continue

        # Demangle for readability
        demangled = demangle(list(dupes.keys()))

        # Filter out known-acceptable duplicates: JUCE module compilation
        # units may define JUCE-internal symbols across their own .mm
        # bundles legitimately. Limit the report to project namespaces.
        project_dupes = {
            s: v for s, v in dupes.items()
            if any(ns in demangled[s] for ns in (
                "penta::", "kelly::", "daiw::", "prrot::"
            ))
        }
        if not project_dupes:
            continue

        total_dupes += len(project_dupes)
        print(f"## `{lib}` — {len(project_dupes)} duplicate text symbols")
        print()
        for sym in sorted(project_dupes, key=lambda s: demangled[s]):
            objs_for_sym = project_dupes[sym]
            print(f"### `{demangled[sym]}`")
            for obj in sorted(objs_for_sym):
                rel = obj.relative_to(BUILD)
                # Map back to source: strip 'CMakeFiles/<lib>.dir/'
                src_hint = str(rel).replace(
                    f"CMakeFiles/{lib}.dir/", "").replace(".cpp.o", ".cpp")
                # Some objects are under '__/__/src/...' (relative includes)
                src_hint = src_hint.replace("__/__/", "../").replace("__/", "../")
                print(f"  - `{src_hint}` (`{rel}`)")
            print()

    if total_dupes == 0:
        print("**No project-namespace intra-library duplicate symbols found.**")
        print()
        print("(JUCE/system internals are ignored; the scanner only flags "
              "duplicates whose demangled names contain `penta::`, `kelly::`, "
              "`daiw::`, or `prrot::`.)")
        return 0

    print(f"---")
    print(f"**Total: {total_dupes} duplicate symbol(s) across project namespaces.**")
    print(f"Each duplicate is a silent ODR violation — the linker picks one "
          f"definition arbitrarily. Fix by ensuring each project symbol is "
          f"defined in exactly one TU.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
