#!/usr/bin/env python3
"""
noexcept-lie scanner for KmiDi's RT engine.

Finds C++ functions declared ``noexcept`` whose bodies allocate, lock,
or throw — each one is a ``bad_alloc``/``lock_error`` away from
``std::terminate``. Per the 2026-04-14 verified audit, this is CRIT
category HIGH.

Usage:
    scripts/audit/noexcept_rt_audit.py > docs/CODEAUDIT_NOEXCEPT_RT_2026Q2.md

Exit code 1 if any CRIT (throw/new/lock) finding exists. Intended to run
in CI once the P0 fixes have landed to prevent regressions.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOTS = ["src", "src_penta-core"]
EXTS = {".cpp"}

NOEXCEPT_DEF = re.compile(
    r"^[\w:<>*&,\s]+[\s&*](\w+)::(\w+)\([^)]*\)\s*(?:const\s*)?noexcept\s*\{",
    re.MULTILINE,
)


def extract_body(text: str, brace_start: int) -> str:
    depth = 0
    i = brace_start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1]
        i += 1
    return text[brace_start:]


def flag_body(body: str) -> list[str]:
    flags = []
    if re.search(r"\bnew\s+[A-Za-z_]", body):
        flags.append("new")
    if re.search(r"\bstd::(lock_guard|unique_lock|scoped_lock)\b|\.lock\(\)", body):
        flags.append("lock")
    if re.search(r"\bthrow\s+[A-Za-z_]", body):
        flags.append("throw")
    if re.search(r"\bstd::to_string\b", body):
        flags.append("to_string")
    if re.search(r"\bstd::string\s+\w+\s*[=;]|\bstd::string\s*\(", body):
        flags.append("string-alloc")
    if re.search(r"\.push_back\(|\.emplace_back\(|\.emplace\(", body):
        if re.search(r"std::(vector|list|deque|map|unordered_map|set)<[^>]*>\s+\w+", body):
            flags.append("container-grow")
    return flags


def severity(flags: list[str]) -> str:
    if any(f in flags for f in ("new", "lock", "throw")):
        return "CRIT"
    if any(f in flags for f in ("string-alloc", "container-grow")):
        return "HIGH"
    return "MED"


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    findings: list[tuple] = []
    for root in ROOTS:
        root_p = repo / root
        if not root_p.is_dir():
            continue
        for p in root_p.rglob("*"):
            if p.suffix not in EXTS:
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            for m in NOEXCEPT_DEF.finditer(text):
                klass, meth = m.group(1), m.group(2)
                line = text[: m.start()].count("\n") + 1
                body = extract_body(text, m.end() - 1)
                flags = flag_body(body)
                if not flags:
                    continue
                rel = p.relative_to(repo)
                findings.append((severity(flags), str(rel), line, klass, meth, flags))

    findings.sort(key=lambda h: (["CRIT", "HIGH", "MED"].index(h[0]), h[1], h[2]))

    print("# noexcept-lie RT Audit — 2026Q2")
    print()
    print(f"Scan produced by `scripts/audit/noexcept_rt_audit.py`. "
          f"{len(findings)} findings across {len(ROOTS)} source roots.")
    print()
    print("Each entry is a function declared `noexcept` whose body allocates, locks, "
          "or throws. In an RT callback context each is one failure away from "
          "`std::terminate`. AU/VST3 validation rejects plugins whose `processBlock` "
          "reaches any of these.")
    print()
    print("**Flag legend**:")
    print("- `new` / `lock` / `throw` → CRIT (program will terminate on failure)")
    print("- `string-alloc` / `container-grow` / `to_string` → HIGH "
          "(implicit heap alloc via libc++; bad_alloc escapes as terminate)")
    print()

    by_sev: dict[str, list] = {"CRIT": [], "HIGH": [], "MED": []}
    for f in findings:
        by_sev[f[0]].append(f)

    for sev in ("CRIT", "HIGH", "MED"):
        if not by_sev[sev]:
            continue
        print(f"## {sev} ({len(by_sev[sev])})")
        print()
        for _, path, line, klass, meth, flags in by_sev[sev]:
            print(f"- `{path}:{line}` — `{klass}::{meth}` "
                  f"[{', '.join(flags)}]")
        print()

    print("## Fix patterns")
    print()
    print("- **`container-grow`**: pre-allocate `std::vector::reserve(max_n)` in "
          "`prepareToPlay`; write via index assignment, never `push_back`. "
          "Or switch to `boost::container::static_vector<T, N>` / "
          "`std::array<T, N>` with a size counter.")
    print("- **`to_string` / `string-alloc`**: replace log lines with "
          "`juce::Logger::writeToLog` gated by `#ifdef JUCE_DEBUG`, or move "
          "the log to the UI thread via a lock-free SPSC queue.")
    print("- **`new`**: allocate in `prepareToPlay`; the RT path only writes "
          "into preallocated buffers.")
    print("- **`lock`**: replace with `std::atomic<T>` or a lock-free SPSC queue "
          "(`readerwriterqueue` is already linked).")
    print("- **`throw`**: return an error code or `std::optional`. RT code "
          "cannot throw across audio callbacks.")
    print()
    print("## CI gate suggestion")
    print()
    print("Once CRIT count is zero, add to the CI workflow:")
    print("```yaml")
    print("- name: noexcept RT audit (regression gate)")
    print("  run: python3 scripts/audit/noexcept_rt_audit.py > /dev/null")
    print("```")
    print()
    print("The script exits non-zero if any CRIT finding exists.")

    return 1 if by_sev["CRIT"] else 0


if __name__ == "__main__":
    sys.exit(main())
