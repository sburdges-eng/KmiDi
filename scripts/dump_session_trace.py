#!/usr/bin/env python3
"""Dump LISTENING_SPEC session transcript records to JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional


def iter_records(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def filter_records(records: Iterable[dict], session_id: Optional[int]) -> list[dict]:
    filtered = []
    for record in records:
        if session_id is not None and int(record.get("sessionId", -1)) != session_id:
            continue
        filtered.append(record)
    filtered.sort(key=lambda item: int(item.get("sequenceId", 0)))
    return filtered


def compute_transcript(records: list[dict], session_id: Optional[int]) -> dict:
    state_sequence: list[str] = []
    terminal_state = "UNKNOWN"
    if records:
        first = records[0]
        if "fromState" in first:
            state_sequence.append(str(first["fromState"]))
        for record in records:
            if "toState" in record:
                state_sequence.append(str(record["toState"]))
        terminal_state = state_sequence[-1] if state_sequence else "UNKNOWN"

    return {
        "kind": "SessionTranscript",
        "sessionId": session_id
        if session_id is not None
        else (records[0].get("sessionId") if records else 0),
        "stateSequence": state_sequence,
        "terminalState": terminal_state,
        "recordCount": len(records),
    }


def write_jsonl(records: list[dict], output_path: Path, session_id: Optional[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                compute_transcript(records, session_id), separators=(",", ":"), sort_keys=True
            )
        )
        handle.write("\n")
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump decision trace records to JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/trace/decision_trace.jsonl"),
        help="Source decision trace JSONL path.",
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=None,
        help="Session ID to filter. If omitted, dumps all sessions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/trace/session_trace_dump.jsonl"),
        help="Output JSONL file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = iter_records(args.input)
    filtered = filter_records(records, args.session_id)
    write_jsonl(filtered, args.output, args.session_id)
    print(f"Wrote {len(filtered)} trace record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
