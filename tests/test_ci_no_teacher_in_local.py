#!/usr/bin/env python3
"""CI Gate G4: No teacher prefix in any local-safe path defaults.

Scans scripts and config files to ensure no argparse default, Path literal,
or YAML/JSON config value points to 'artifacts/teacher' as a local destination.

This check is deliberately narrow: it only flags lines where a teacher path
is used as a *default value* or *config literal*. Guards, error messages,
assertions, and conditional checks that reference teacher paths are fine.

Exit code 0 = pass, 1 = fail.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(".")

SCAN_GLOBS = [
    "scripts/*.py",
    "config/*.yaml",
    "config/*.json",
]

# Only flag lines that assign a teacher path as a default or literal destination.
TEACHER_LOCAL_PATTERNS = [
    # argparse default pointing to teacher artifact path
    re.compile(r"""default\s*=\s*["'].*artifacts/teacher""", re.IGNORECASE),
    re.compile(r"""default\s*=\s*Path\(["'].*artifacts/teacher""", re.IGNORECASE),
    # YAML/JSON config values that set a *local directory* to teacher
    re.compile(
        r"""(outputDir|output_dir|downloadDir|download_dir|localDir|local_dir)"""
        r"""\s*:\s*["'].*artifacts/teacher""",
        re.IGNORECASE,
    ),
    # Variable assignment of a local path to teacher
    re.compile(r"""output_dir\s*=.*artifacts/teacher""", re.IGNORECASE),
    re.compile(r"""download_dir\s*=.*artifacts/teacher""", re.IGNORECASE),
    re.compile(r"""local_path\s*=.*artifacts/teacher""", re.IGNORECASE),
]

# Lines to always skip
SKIP_PATTERNS = [
    re.compile(r"^\s*#"),  # comments
    re.compile(r'^\s*"""'),  # docstrings
    re.compile(r"^\s*'"),  # single-quote docstrings
    re.compile(r"raise\s"),  # error raising
    re.compile(r"assert"),  # assertions
    re.compile(r"if\s.*teacher"),  # conditional guards
    re.compile(r"startswith\("),  # prefix checks
    re.compile(r"prohibit", re.IGNORECASE),
]


def should_skip_line(line: str) -> bool:
    return any(p.search(line) for p in SKIP_PATTERNS)


def scan_file(path: Path) -> list[str]:
    violations: list[str] = []
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return violations

    for line_num, line in enumerate(content.splitlines(), 1):
        if should_skip_line(line):
            continue
        for pattern in TEACHER_LOCAL_PATTERNS:
            if pattern.search(line):
                violations.append(f"  {path}:{line_num}: {line.strip()}")
                break  # one violation per line
    return violations


def main() -> int:
    all_violations: list[str] = []

    for glob_pattern in SCAN_GLOBS:
        for path in sorted(REPO_ROOT.glob(glob_pattern)):
            all_violations.extend(scan_file(path))

    if all_violations:
        print("FAIL [G4]: Teacher prefix found in local-safe path defaults:")
        for v in all_violations:
            print(v)
        return 1

    print("PASS [G4]: No teacher prefix in local-safe path defaults")
    return 0


if __name__ == "__main__":
    sys.exit(main())
