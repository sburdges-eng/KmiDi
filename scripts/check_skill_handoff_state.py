#!/usr/bin/env python3
"""Check that shared handoff-state JSON blocks are identical across skill runbooks."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

SECTION_HEADING = "## Shared Handoff State Format"
EXPECTED_KEYS = ["package-id", "runId", "instanceId", "sessionName", "outputS3Uri"]
DEFAULT_RUNBOOKS = (
    "skills/dataset-packaging/references/dataset-packaging-runbook.md",
    "skills/training-ops-cockpit/references/training-ops-runbook.md",
    "skills/training-pipeline-orchestrator/references/pipeline-orchestration-runbook.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify shared handoff-state JSON block consistency across skill runbooks."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root path (defaults to script parent).",
    )
    parser.add_argument(
        "--runbook",
        action="append",
        default=[],
        help="Runbook path relative to --repo-root. Repeatable. Defaults to known runbooks.",
    )
    return parser.parse_args()


def normalize_block(block: str) -> str:
    lines = [line.rstrip() for line in block.strip().splitlines()]
    return "\n".join(lines)


def extract_handoff_json_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find(SECTION_HEADING)
    if start < 0:
        raise ValueError(f"Missing section heading '{SECTION_HEADING}'")

    tail = text[start:]
    match = re.search(r"```json\s*\n(.*?)\n```", tail, flags=re.DOTALL)
    if match is None:
        raise ValueError("Missing fenced ```json block after shared handoff heading")
    return normalize_block(match.group(1))


def validate_shape(block: str, path: Path) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON in shared handoff block ({exc})"]

    if not isinstance(payload, dict):
        return [f"{path}: shared handoff JSON must be an object"]

    keys = list(payload.keys())
    if keys != EXPECTED_KEYS:
        errors.append(f"{path}: expected keys/order {EXPECTED_KEYS}, got {keys}")

    for key in EXPECTED_KEYS:
        value = payload.get(key)
        if value != "":
            errors.append(f"{path}: expected empty string for key '{key}', got {value!r}")
    return errors


def emit_diff(base_path: Path, base_block: str, path: Path, block: str) -> None:
    diff = difflib.unified_diff(
        base_block.splitlines(),
        block.splitlines(),
        fromfile=str(base_path),
        tofile=str(path),
        lineterm="",
    )
    print("\n".join(diff))


def main() -> int:
    args = parse_args()
    runbooks = tuple(args.runbook) if args.runbook else DEFAULT_RUNBOOKS

    if len(runbooks) < 2:
        raise ValueError("Provide at least two --runbook values (or use defaults)")

    resolved_paths = [args.repo_root / rel_path for rel_path in runbooks]
    blocks: dict[Path, str] = {}
    failures: list[str] = []

    for path in resolved_paths:
        if not path.exists():
            failures.append(f"Missing runbook file: {path}")
            continue
        try:
            block = extract_handoff_json_block(path)
        except ValueError as exc:
            failures.append(f"{path}: {exc}")
            continue

        failures.extend(validate_shape(block, path))
        blocks[path] = block

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1

    base_path = resolved_paths[0]
    base_block = blocks[base_path]
    mismatch = False
    for path in resolved_paths[1:]:
        block = blocks[path]
        if block != base_block:
            mismatch = True
            print(f"MISMATCH: shared handoff block differs in {path}")
            emit_diff(base_path, base_block, path, block)

    if mismatch:
        return 1

    print(f"OK: shared handoff block is consistent across {len(resolved_paths)} runbooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
