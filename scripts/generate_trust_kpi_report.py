#!/usr/bin/env python3
"""Generate trust KPI report.

KPI definition:
  % of interactions where the system did exactly what was asked, and nothing else.

Input JSONL schema (one object per line):
  {
    "id": "optional-string",
    "channel": "optional-string (text|parameter|performance|mixed)",
    "exact_match": true|false,   # required for scored rows
    "abstained": true|false,     # optional
    "violation": true|false,     # optional
    "notes": "optional"
  }
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Row:
    channel: str
    exact_match: bool
    abstained: bool
    violation: bool


def parse_rows(path: Path) -> tuple[list[Row], int]:
    rows: list[Row] = []
    skipped = 0
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if "exact_match" not in obj or not isinstance(obj["exact_match"], bool):
            skipped += 1
            continue
        channel = obj.get("channel", "unspecified")
        if not isinstance(channel, str) or not channel:
            channel = "unspecified"
        abstained = bool(obj.get("abstained", False))
        violation = bool(obj.get("violation", False))
        rows.append(
            Row(
                channel=channel,
                exact_match=obj["exact_match"],
                abstained=abstained,
                violation=violation,
            )
        )
    return rows, skipped


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def build_report(rows: list[Row], skipped: int, min_sample: int, window_label: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    total = len(rows)
    exact = sum(1 for r in rows if r.exact_match)
    abstain = sum(1 for r in rows if r.abstained)
    violations = sum(1 for r in rows if r.violation)

    by_channel: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_channel[r.channel].append(r)

    lines: list[str] = []
    lines.append("# Trust KPI Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {ts}")
    lines.append(f"- Window: {window_label}")
    lines.append(
        "- KPI: % interactions where output matched request exactly, with no unrequested changes"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if total == 0:
        lines.append("- Status: INSUFFICIENT_DATA")
        lines.append("- Sample size: 0")
        lines.append("- Trust KPI: N/A")
    else:
        lines.append(f"- Sample size: {total}")
        lines.append(f"- Exact matches: {exact}")
        lines.append(f"- Trust KPI: {pct(exact, total):.2f}%")
        lines.append(f"- Abstain rate: {pct(abstain, total):.2f}% ({abstain}/{total})")
        lines.append(f"- Violation rate: {pct(violations, total):.2f}% ({violations}/{total})")
        if total < min_sample:
            lines.append(f"- Data quality: LOW_CONFIDENCE (sample < {min_sample})")
        else:
            lines.append(f"- Data quality: OK (sample >= {min_sample})")
    if skipped:
        lines.append(f"- Skipped rows (invalid schema): {skipped}")

    lines.append("")
    lines.append("## Channel Breakdown")
    lines.append("")
    lines.append("| Channel | Samples | Exact | KPI % | Abstain % | Violation % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for channel in sorted(by_channel.keys()):
        bucket = by_channel[channel]
        n = len(bucket)
        e = sum(1 for r in bucket if r.exact_match)
        a = sum(1 for r in bucket if r.abstained)
        v = sum(1 for r in bucket if r.violation)
        lines.append(
            f"| {channel} | {n} | {e} | {pct(e, n):.2f} | {pct(a, n):.2f} | {pct(v, n):.2f} |"
        )
    if not by_channel:
        lines.append("| (none) | 0 | 0 | 0.00 | 0.00 | 0.00 |")

    lines.append("")
    lines.append("## Sign-off Guidance")
    lines.append("")
    if total == 0:
        lines.append("- Provide an interaction assertion dataset before release sign-off.")
    else:
        lines.append(
            "- Use this KPI with guardrail/runtime evidence; do not sign off on KPI alone."
        )
        lines.append("- Investigate any violation > 0 before final release tag.")
        if total < min_sample:
            lines.append("- Increase sample size for stronger confidence before RC tag.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate trust KPI markdown report from JSONL interactions."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Path to interaction assertions JSONL"
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to markdown report output")
    parser.add_argument(
        "--min-sample", type=int, default=20, help="Minimum sample size for confidence"
    )
    parser.add_argument(
        "--window-label", default="release-window", help="Window label shown in report"
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        report = build_report(
            [], skipped=0, min_sample=args.min_sample, window_label=args.window_label
        )
        report += "\nInput file not found: " + str(args.input) + "\n"
        args.output.write_text(report)
        return

    rows, skipped = parse_rows(args.input)
    report = build_report(
        rows, skipped=skipped, min_sample=args.min_sample, window_label=args.window_label
    )
    args.output.write_text(report)


if __name__ == "__main__":
    main()
