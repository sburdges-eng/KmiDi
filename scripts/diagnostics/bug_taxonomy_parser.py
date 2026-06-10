#!/usr/bin/env python3
"""
Deterministic classifier for baseline debug logs.

This script scans log files and categorizes findings using the requested bug taxonomy.
It intentionally produces stable ordering to support reproducible reviews.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


BUG_CATEGORIES: List[str] = [
    "Logical bugs",
    "Syntax bugs",
    "Runtime errors",
    "Semantic bugs",
    "Concurrency bugs",
    "Race conditions",
    "Deadlocks",
    "Off-by-one errors",
    "Performance bugs",
    "Memory leaks",
    "Null pointer errors",
    "Type errors",
    "Overflow errors",
    "Underflow errors",
    "Floating point precision errors",
    "Security vulnerabilities",
    "Injection flaws",
    "Authentication bugs",
    "Authorization bugs",
    "Integration bugs",
    "API contract violations",
    "Data corruption bugs",
    "Encoding/decoding bugs",
    "State machine bugs",
    "Initialization bugs",
    "Configuration bugs",
    "Dependency bugs",
    "Build/compile errors",
    "Regression bugs",
    "Edge case bugs",
    "Boundary condition bugs",
    "Resource exhaustion bugs",
    "Timeout errors",
    "Exception handling bugs",
    "Infinite loop bugs",
    "Recursion overflow bugs",
    "UI rendering bugs",
    "Event handling bugs",
    "Caching bugs",
    "Serialization bugs",
    "Deserialization bugs",
]

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
WHITESPACE = re.compile(r"\s+")
ABS_PATH = re.compile(r"/(?:Users|Volumes|private|System|usr|opt|Library|Applications)(?:/[A-Za-z0-9._:+-]+)+")
ACTIONABLE_SIGNAL = re.compile(
    r"\berror\b|"
    r"\bfatal\b|"
    r"\bfailed\b|"
    r"\bwarning\b|"
    r"\btraceback\b|"
    r"\bassertionerror\b|"
    r"\bimporterror\b|"
    r"no tests were found|"
    r"command not found|"
    r"cannot import|"
    r"could not resolve host|"
    r"no such file or directory",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    categories: Tuple[str, ...]


@dataclass
class Finding:
    log_file: str
    line_no: int
    message: str
    categories: Set[str] = field(default_factory=set)

    def merge_categories(self, extra: Iterable[str]) -> None:
        self.categories.update(extra)


RULES: List[Rule] = [
    Rule(re.compile(r"\bSyntaxError\b|parse error|unexpected token", re.IGNORECASE), ("Syntax bugs", "Build/compile errors")),
    Rule(re.compile(r"\bImportError\b|cannot import name|partially initialized module", re.IGNORECASE), ("Runtime errors", "Initialization bugs", "Integration bugs")),
    Rule(re.compile(r"\bTraceback\b|unhandled exception|\bException\b", re.IGNORECASE), ("Runtime errors", "Exception handling bugs")),
    Rule(re.compile(r"\bAssertionError\b|assert false|assertion failed", re.IGNORECASE), ("Logical bugs", "Semantic bugs", "Regression bugs")),
    Rule(re.compile(r"\bTypeError\b|type mismatch|cannot convert", re.IGNORECASE), ("Type errors", "Runtime errors")),
    Rule(re.compile(r"NoneType|None\b|null pointer|nullptr", re.IGNORECASE), ("Null pointer errors", "Edge case bugs")),
    Rule(re.compile(r"index out of range|out of bounds|off[- ]by[- ]one", re.IGNORECASE), ("Off-by-one errors", "Boundary condition bugs")),
    Rule(re.compile(r"\boverflow\b|integer too large", re.IGNORECASE), ("Overflow errors",)),
    Rule(re.compile(r"\bunderflow\b", re.IGNORECASE), ("Underflow errors",)),
    Rule(re.compile(r"floating point|precision|NaN|\\binf\\b", re.IGNORECASE), ("Floating point precision errors",)),
    Rule(re.compile(r"\bdeadlock\b", re.IGNORECASE), ("Deadlocks", "Concurrency bugs")),
    Rule(re.compile(r"\brace\b|data race", re.IGNORECASE), ("Race conditions", "Concurrency bugs")),
    Rule(re.compile(r"\bthread\b|\bmutex\b|\block\b", re.IGNORECASE), ("Concurrency bugs",)),
    Rule(re.compile(r"slow|latency|performance|hot path", re.IGNORECASE), ("Performance bugs",)),
    Rule(re.compile(r"memory leak|leak sanitizer|AddressSanitizer: detected memory leaks", re.IGNORECASE), ("Memory leaks", "Resource exhaustion bugs")),
    Rule(re.compile(r"resource exhausted|out of memory|too many open files|disk full", re.IGNORECASE), ("Resource exhaustion bugs",)),
    Rule(re.compile(r"\btimeout\b|timed out", re.IGNORECASE), ("Timeout errors",)),
    Rule(re.compile(r"\binfinite loop\b", re.IGNORECASE), ("Infinite loop bugs",)),
    Rule(re.compile(r"maximum recursion depth|stack overflow", re.IGNORECASE), ("Recursion overflow bugs",)),
    Rule(re.compile(r"\bserialize\b|\bserialization\b", re.IGNORECASE), ("Serialization bugs",)),
    Rule(re.compile(r"\bdeserialize\b|\bdeserialization\b", re.IGNORECASE), ("Deserialization bugs",)),
    Rule(re.compile(r"decode|encode|UnicodeDecodeError|UnicodeEncodeError", re.IGNORECASE), ("Encoding/decoding bugs",)),
    Rule(re.compile(r"corrupt|checksum mismatch|invalid magic|data loss", re.IGNORECASE), ("Data corruption bugs",)),
    Rule(re.compile(r"invalid transition|invalid state|state transition", re.IGNORECASE), ("State machine bugs", "Semantic bugs")),
    Rule(re.compile(r"security|vulnerab|CVE", re.IGNORECASE), ("Security vulnerabilities",)),
    Rule(re.compile(r"injection|sql syntax|xss|command injection", re.IGNORECASE), ("Injection flaws", "Security vulnerabilities")),
    Rule(re.compile(r"authentication|invalid credential|login failed", re.IGNORECASE), ("Authentication bugs", "Security vulnerabilities")),
    Rule(re.compile(r"authorization|forbidden|permission denied|not allowed", re.IGNORECASE), ("Authorization bugs", "Security vulnerabilities")),
    Rule(re.compile(r"api contract|schema mismatch|missing required|unexpected field|status code", re.IGNORECASE), ("API contract violations", "Integration bugs")),
    Rule(re.compile(r"\bui\b|\brender(?:ing)?\b|\blayout\b|\bpaint\b", re.IGNORECASE), ("UI rendering bugs",)),
    Rule(re.compile(r"event handler|callback|onClick|onChange", re.IGNORECASE), ("Event handling bugs",)),
    Rule(re.compile(r"\bcache\b|stale cache|cache miss", re.IGNORECASE), ("Caching bugs",)),
    Rule(re.compile(r"\bboundary\b|\bclamp(?:ing)?\b|\bminimum\b|\bmaximum\b|\bout of range\b|\bmin\b|\bmax\b", re.IGNORECASE), ("Boundary condition bugs",)),
    Rule(re.compile(r"\bedge case\b|\btest_none\b|\bnone inputs\b", re.IGNORECASE), ("Edge case bugs",)),
    Rule(re.compile(r"CMake Error|ninja: error|build stopped|compile error", re.IGNORECASE), ("Build/compile errors",)),
    Rule(re.compile(r"command not found|Could NOT find|No such file or directory|No tests were found", re.IGNORECASE), ("Configuration bugs",)),
    Rule(re.compile(r"failed to get `|FetchContent|git clone|dependency|module not found", re.IGNORECASE), ("Dependency bugs",)),
    Rule(re.compile(r"no matching package named|crates\\.io index|offline mode", re.IGNORECASE), ("Dependency bugs", "Configuration bugs")),
    Rule(re.compile(r"Could not resolve host|failed to download|unable to access 'https?://|connection refused", re.IGNORECASE), ("Dependency bugs", "Integration bugs", "Configuration bugs")),
]


def normalize_message(raw: str) -> str:
    clean = ANSI_ESCAPE.sub("", raw).strip()
    clean = WHITESPACE.sub(" ", clean)
    clean = ABS_PATH.sub("<ABS_PATH>", clean)
    return clean


def classify_line(line: str) -> Set[str]:
    categories: Set[str] = set()
    for rule in RULES:
        if rule.pattern.search(line):
            categories.update(rule.categories)
    return categories


def parse_log(path: Path) -> Dict[Tuple[str, str], Finding]:
    findings: Dict[Tuple[str, str], Finding] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        message = normalize_message(raw_line)
        if not message:
            continue
        categories = classify_line(message)
        if not categories:
            continue
        if not ACTIONABLE_SIGNAL.search(message) and not message.startswith("E + where"):
            continue
        key = (path.name, message)
        if key not in findings:
            findings[key] = Finding(
                log_file=path.name,
                line_no=line_no,
                message=message,
                categories=set(categories),
            )
        else:
            findings[key].merge_categories(categories)
    return findings


def parse_status(status_path: Path) -> List[Tuple[str, str]]:
    if not status_path.exists():
        return []
    rows: List[Tuple[str, str]] = []
    end_pattern = re.compile(r"END\s+([a-zA-Z0-9_-]+)\s+rc=(\d+)")
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = normalize_message(line)
        match = end_pattern.search(clean)
        if match:
            rows.append((match.group(1), match.group(2)))
    return sorted(rows, key=lambda item: item[0])


def count_categories(findings: List[Finding]) -> Dict[str, int]:
    counts = {category: 0 for category in BUG_CATEGORIES}
    for finding in findings:
        for category in sorted(finding.categories):
            if category in counts:
                counts[category] += 1
    return counts


def render_counts_file(counts: Dict[str, int], generated_at: str) -> str:
    """Baseline-format counts (<Category>=<Count>), commit-ready for the CI gate."""
    lines = [
        "# Baseline category counts for CI gate.",
        "# Format: <Category>=<Count>",
        f"# Generated by bug_taxonomy_parser.py at {generated_at}.",
        "",
    ]
    lines.extend(f"{category}={counts[category]}" for category in BUG_CATEGORIES)
    return "\n".join(lines) + "\n"


def render_report(
    logs_dir: Path,
    findings: List[Finding],
    status_rows: List[Tuple[str, str]],
    generated_at: str,
) -> str:
    counts = count_categories(findings)

    lines: List[str] = []
    lines.append("# Bug Taxonomy Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{generated_at}`")
    lines.append(f"- Logs directory: `{logs_dir}`")
    lines.append(f"- Finding count: `{len(findings)}`")
    lines.append("")
    lines.append("## Command Status")
    lines.append("")
    if not status_rows:
        lines.append("_No status file found._")
    else:
        lines.append("| Command | Exit Code |")
        lines.append("|---|---:|")
        for command, exit_code in status_rows:
            lines.append(f"| `{command}` | `{exit_code}` |")
    lines.append("")
    lines.append("## Category Counts")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for category in BUG_CATEGORIES:
        lines.append(f"| {category} | {counts[category]} |")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not findings:
        lines.append("_No classified findings detected in the provided logs._")
    else:
        for index, finding in enumerate(findings, start=1):
            categories = ", ".join(sorted(finding.categories))
            lines.append(
                f"{index}. `{finding.log_file}:{finding.line_no}` | "
                f"Categories: {categories} | Evidence: `{finding.message}`"
            )
    lines.append("")
    lines.append("## Determinism Notes")
    lines.append("")
    lines.append("- Report ordering is stable by `(log file, line number, message)`.")
    lines.append("- Classification is rule-based and does not depend on external services.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify baseline logs into bug taxonomy categories.")
    parser.add_argument(
        "--logs-dir",
        default="artifacts/debug_baseline",
        help="Directory containing baseline .log files",
    )
    parser.add_argument(
        "--out",
        default="artifacts/debug_baseline/bug_taxonomy_report.md",
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--counts-out",
        default=None,
        help="Optional path for baseline-format category counts (<Category>=<Count>)",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_path = Path(args.out)

    if not logs_dir.exists():
        raise SystemExit(f"Logs directory does not exist: {logs_dir}")

    log_paths = sorted(logs_dir.glob("*.log"), key=lambda p: p.name)
    finding_map: Dict[Tuple[str, str], Finding] = {}
    for log_path in log_paths:
        finding_map.update(parse_log(log_path))

    findings = sorted(
        finding_map.values(),
        key=lambda f: (f.log_file, f.line_no, f.message),
    )
    status_rows = parse_status(logs_dir / "status.txt")
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = render_report(logs_dir, findings, status_rows, generated_at)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote report: {output_path}")
    if args.counts_out:
        counts_path = Path(args.counts_out)
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        counts_path.write_text(
            render_counts_file(count_categories(findings), generated_at),
            encoding="utf-8",
        )
        print(f"Wrote counts: {counts_path}")
    print(f"Findings: {len(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
