#!/usr/bin/env python3
"""Compute sequence-length histograms for REMI vs REMI+BPE to validate vocab size and length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_token_lengths(path: Path) -> list[int]:
    """Load token IDs from a JSON file (one seq per line or single list of token IDs)."""
    lengths = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, list):
                lengths.append(len(data))
            elif isinstance(data, dict) and "ids" in data:
                lengths.append(len(data["ids"]))
            elif isinstance(data, dict) and "token_ids" in data:
                lengths.append(len(data["token_ids"]))
            else:
                lengths.append(len(data) if hasattr(data, "__len__") else 0)
    return lengths


def collect_lengths_from_dir(dir_path: Path, suffix: str = ".json") -> list[int]:
    lengths = []
    for f in dir_path.rglob(f"*{suffix}"):
        if f.is_file():
            lengths.extend(load_token_lengths(f))
    return lengths


def histogram_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "median": 0, "mean": 0.0, "p90": 0, "p99": 0}
    sorted_len = sorted(lengths)
    n = len(sorted_len)
    return {
        "count": n,
        "min": sorted_len[0],
        "max": sorted_len[-1],
        "median": sorted_len[n // 2],
        "mean": sum(lengths) / n,
        "p90": sorted_len[int(n * 0.90)] if n >= 10 else sorted_len[-1],
        "p99": sorted_len[int(n * 0.99)] if n >= 100 else sorted_len[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute seq-length stats for tokenized MIDI (REMI or REMI+BPE)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories containing tokenized JSON "
        "(one seq per line or 'ids'/'token_ids' key).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write stats JSON here.",
    )
    parser.add_argument(
        "--suffix",
        default=".json",
        help="File suffix when paths are dirs.",
    )
    args = parser.parse_args()

    all_lengths = []
    for p in args.paths:
        if p.is_dir():
            all_lengths.extend(collect_lengths_from_dir(p, suffix=args.suffix))
        elif p.is_file():
            all_lengths.extend(load_token_lengths(p))

    stats = histogram_stats(all_lengths)
    out = {
        "description": "Sequence length stats (median under ~1k is target for 1024 window).",
        "stats": stats,
    }
    print(json.dumps(out, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
