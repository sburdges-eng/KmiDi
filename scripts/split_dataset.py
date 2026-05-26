#!/usr/bin/env python3
"""Deterministically split task records by sessionId hash."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.training_common import choose_split, iter_jsonl, utc_now_iso, write_json, write_jsonl


TASK_INPUTS = {
    "intent_router": "intent_router_records.jsonl",
    "axis_proposer": "axis_proposer_records.jsonl",
}


def record_sort_key(record: dict[str, Any]) -> tuple[str, str, int]:
    session_id = str(record.get("sessionId", ""))
    sample_id = str(record.get("sampleId", ""))
    source_line = int(record.get("sourceLine", 0) or 0)
    return session_id, sample_id, source_line


def split_task_records(
    records: list[dict[str, Any]],
    seed: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    ratio_map = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    split_map: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for record in records:
        session_id = str(record.get("sessionId", ""))
        if not session_id:
            continue
        split_name = choose_split(session_id, seed, ratio_map)
        split_map[split_name].append(record)

    for split_name in split_map:
        split_map[split_name].sort(key=record_sort_key)
    return split_map


def write_task_splits(
    task: str, split_map: dict[str, list[dict[str, Any]]], output_root: Path
) -> dict[str, int]:
    task_dir = output_root / task
    task_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split_name, records in split_map.items():
        output_path = task_dir / f"{split_name}.jsonl"
        counts[split_name] = write_jsonl(output_path, records)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministically split datasets by sessionId.")
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=Path("training/records"),
        help="Directory containing *_records.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/splits"),
        help="Directory to write split JSONL files.",
    )
    parser.add_argument("--seed", default="listening-spec-v1", help="Stable split seed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to > 0")

    summary: dict[str, Any] = {
        "generatedAt": utc_now_iso(),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "tasks": {},
    }

    for task, filename in TASK_INPUTS.items():
        input_path = args.records_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Missing task input: {input_path}")

        records = list(iter_jsonl(input_path))
        split_map = split_task_records(
            records,
            args.seed,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        counts = write_task_splits(task, split_map, args.output_dir)
        summary["tasks"][task] = {
            "input": str(input_path),
            "counts": counts,
            "total": len(records),
        }
        print(
            f"{task}: train={counts['train']} val={counts['val']} "
            f"test={counts['test']} (total={len(records)})"
        )

    summary_path = args.output_dir / "split_summary.json"
    write_json(summary_path, summary)
    print(f"Wrote split summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
