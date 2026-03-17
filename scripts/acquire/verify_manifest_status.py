#!/usr/bin/env python3
"""
Report verification and adoption status for config/source_manifest.yaml (Phase 5).

Read-only: no downloads, no manifest edits. Supports Stage 1 — Verify first.

Usage:
    python scripts/acquire/verify_manifest_status.py
    python scripts/acquire/verify_manifest_status.py --check-urls
    python scripts/acquire/verify_manifest_status.py --json
    python scripts/acquire/verify_manifest_status.py --strict

See docs/SOURCE_INTEGRATION_PLAN.md §9 Phase 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_SOURCE_MANIFEST = REPO_ROOT / "config" / "source_manifest.yaml"

# Truncate URL for table display
URL_MAX = 52


def _load_manifest(path: Path) -> list[dict]:
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml required; pip install pyyaml")
    if not path.is_file():
        sys.exit(f"Manifest not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("sources") or []


def _head_url(url: str, timeout: int = 10) -> str:
    """HEAD request; return 'ok', 'fail', 'redirect', 'timeout', or 'skip'."""
    if not url or url.upper() in ("UNKNOWN", "N/A", "NULL") or not url.startswith(("http://", "https://")):
        return "skip"
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "KmiDi-verify-manifest/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 400:
                return "ok"
            if 300 <= resp.status < 400:
                return "redirect"
            return "fail"
    except TimeoutError:
        return "timeout"
    except Exception:
        return "fail"


def _truncate(s: str | None, max_len: int = URL_MAX) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return s if len(s) <= max_len else s[: max_len - 2] + ".."


def run_report(manifest_path: Path, check_urls: bool, as_json: bool, strict: bool) -> int:
    sources = _load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []

    for item in sources:
        source_item = item.get("source_item") or "?"
        verification_status = item.get("verification_status") or ""
        license_val = item.get("license") or ""
        adoption_decision = item.get("adoption_decision") or ""
        downloadable = item.get("downloadable") or ""
        primary_url = item.get("primary_url") or ""

        url_status = ""
        if check_urls and primary_url:
            url_status = _head_url(primary_url)

        row = {
            "source_item": source_item,
            "verification_status": verification_status,
            "license": license_val,
            "adoption_decision": adoption_decision,
            "downloadable": downloadable,
            "primary_url": _truncate(primary_url, 200) if as_json else _truncate(primary_url, URL_MAX),
        }
        if check_urls:
            row["url_status"] = url_status
        rows.append(row)

    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        # Table header
        cols = ["source_item", "verification_status", "license", "adoption_decision", "downloadable", "primary_url"]
        if check_urls:
            cols.append("url_status")
        widths = [20, 18, 12, 14, 12, URL_MAX + 2]
        if check_urls:
            widths.append(10)
        header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
        print(header)
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for r in rows:
            parts = [
                (r["source_item"], widths[0]),
                (r["verification_status"], widths[1]),
                (r["license"], widths[2]),
                (r["adoption_decision"], widths[3]),
                (r["downloadable"], widths[4]),
                (r["primary_url"], widths[5]),
            ]
            if check_urls:
                parts.append((r.get("url_status", ""), widths[6]))
            print("  ".join(str(p[0])[: p[1]].ljust(p[1]) for p in parts))

    if strict:
        # Red flag: downloadable and license UNKNOWN
        for r in rows:
            if (r["downloadable"] in ("yes", "unknown")) and (r["license"] in ("UNKNOWN", "")):
                print(f"\n[strict] Red flag: {r['source_item']} has downloadable={r['downloadable']} and license={r['license']}", file=sys.stderr)
                return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Report source_manifest verification and adoption status (read-only).",
    )
    ap.add_argument("--manifest", type=Path, default=CONFIG_SOURCE_MANIFEST, help="Path to source_manifest.yaml")
    ap.add_argument("--check-urls", action="store_true", help="HEAD primary_url for each entry (no download)")
    ap.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if any downloadable item has license UNKNOWN")
    args = ap.parse_args()
    return run_report(args.manifest, args.check_urls, args.json, args.strict)


if __name__ == "__main__":
    sys.exit(main())
