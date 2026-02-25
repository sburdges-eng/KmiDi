#!/usr/bin/env python3
"""
CI gate for bug taxonomy regressions.

Compares category counts in a current report against a baseline.
Fails when counts increase for categories at or above the configured severity.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}

HIGH_SEVERITY = {
    "Security vulnerabilities",
    "Injection flaws",
    "Authentication bugs",
    "Authorization bugs",
    "Data corruption bugs",
    "Deadlocks",
    "Race conditions",
    "Concurrency bugs",
    "Resource exhaustion bugs",
    "Runtime errors",
    "Build/compile errors",
    "API contract violations",
    "Regression bugs",
    "Null pointer errors",
}

MEDIUM_SEVERITY = {
    "Logical bugs",
    "Syntax bugs",
    "Semantic bugs",
    "Off-by-one errors",
    "Performance bugs",
    "Memory leaks",
    "Type errors",
    "Overflow errors",
    "Underflow errors",
    "Floating point precision errors",
    "Integration bugs",
    "Encoding/decoding bugs",
    "State machine bugs",
    "Initialization bugs",
    "Configuration bugs",
    "Dependency bugs",
    "Boundary condition bugs",
    "Timeout errors",
    "Exception handling bugs",
    "Infinite loop bugs",
    "Recursion overflow bugs",
    "UI rendering bugs",
    "Event handling bugs",
    "Caching bugs",
    "Serialization bugs",
    "Deserialization bugs",
}


def severity_for(category: str) -> str:
    if category in HIGH_SEVERITY:
        return "high"
    if category in MEDIUM_SEVERITY:
        return "medium"
    return "low"


def parse_report_counts(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    in_table = False
    row_re = re.compile(r"^\|\s*(.+?)\s*\|\s*(\d+)\s*\|\s*$")
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "## Category Counts":
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("## ") and line != "## Category Counts":
            break
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            continue
        match = row_re.match(line)
        if match:
            category = match.group(1)
            count = int(match.group(2))
            if category != "Category":
                counts[category] = count
    return counts


def parse_baseline(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        category, raw_count = line.split("=", 1)
        category = category.strip()
        try:
            counts[category] = int(raw_count.strip())
        except ValueError:
            continue
    return counts


def compare_counts(
    baseline: Dict[str, int],
    current: Dict[str, int],
) -> List[Tuple[str, int, int, int, str]]:
    rows: List[Tuple[str, int, int, int, str]] = []
    all_categories = sorted(set(baseline) | set(current))
    for category in all_categories:
        base_count = baseline.get(category, 0)
        curr_count = current.get(category, 0)
        delta = curr_count - base_count
        rows.append((category, base_count, curr_count, delta, severity_for(category)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail CI on bug taxonomy regressions.")
    parser.add_argument("--baseline", required=True, help="Baseline counts file")
    parser.add_argument("--current-report", required=True, help="Current markdown report file")
    parser.add_argument(
        "--min-severity",
        choices=("low", "medium", "high"),
        default="medium",
        help="Minimum severity that triggers a failure when increased",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    report_path = Path(args.current_report)
    if not baseline_path.exists():
        raise SystemExit(f"Baseline file missing: {baseline_path}")
    if not report_path.exists():
        raise SystemExit(f"Report file missing: {report_path}")

    baseline_counts = parse_baseline(baseline_path)
    current_counts = parse_report_counts(report_path)
    rows = compare_counts(baseline_counts, current_counts)

    threshold = SEVERITY_ORDER[args.min_severity]
    regressions = [
        row for row in rows if row[3] > 0 and SEVERITY_ORDER[row[4]] >= threshold
    ]

    print(f"Baseline: {baseline_path}")
    print(f"Current : {report_path}")
    print(f"Threshold severity: {args.min_severity}")
    print("")
    print("Category delta summary:")
    for category, base_count, curr_count, delta, severity in rows:
        if delta != 0:
            print(f"- {category}: baseline={base_count}, current={curr_count}, delta={delta:+d}, severity={severity}")

    if regressions:
        print("")
        print("CI gate failed: severity-threshold regressions detected.")
        for category, base_count, curr_count, delta, severity in regressions:
            print(f"* {category}: baseline={base_count}, current={curr_count}, delta={delta:+d}, severity={severity}")
        return 1

    print("")
    print("CI gate passed: no severity-threshold regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
