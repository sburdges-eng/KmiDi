#!/usr/bin/env python3
"""Build task-specific training records from transcript JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.training_common import (
    INTENT_LABELS,
    int_or_zero,
    iter_jsonl,
    normalize_boolish,
    utc_now_iso,
    write_json,
)


def load_axes_config(path: Path) -> tuple[list[str], list[str]]:
    text = path.read_text(encoding="utf-8")
    payload: dict[str, Any]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"Unable to parse {path}: install PyYAML or provide JSON-compatible YAML"
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid axes config in {path}")
        payload = loaded

    axes = payload.get("axes", [])
    if not isinstance(axes, list) or not axes:
        raise ValueError("config/axes.yaml must define non-empty 'axes' list")
    axis_names = [str(item) for item in axes if str(item).strip()]
    if not axis_names:
        raise ValueError("Axis names cannot be empty")

    candidates = payload.get("axisFieldCandidates", ["axisTargets", "axes", "targets"])
    if not isinstance(candidates, list) or not candidates:
        candidates = ["axisTargets", "axes", "targets"]
    field_candidates = [str(item) for item in candidates if str(item).strip()]
    return axis_names, field_candidates


def extract_session_id(record: dict[str, Any]) -> str:
    direct = record.get("sessionId")
    if direct is not None:
        return str(direct)

    session = record.get("session")
    if isinstance(session, dict):
        sid = session.get("id") or session.get("sessionId")
        if sid is not None:
            return str(sid)

    return ""


def extract_text(record: dict[str, Any]) -> str:
    candidate_keys = [
        "inputText",
        "text",
        "prompt",
        "request",
        "utterance",
        "sideBInput",
        "input",
    ]
    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("text") or value.get("prompt")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def extract_intent_label(record: dict[str, Any]) -> str:
    candidates = [
        record.get("decision"),
        record.get("intentResult"),
        record.get("intentStatus"),
        record.get("status"),
    ]

    intent = record.get("intent")
    if isinstance(intent, dict):
        candidates.append(intent.get("status"))
        candidates.append(intent.get("decision"))

    for raw in candidates:
        if raw is None:
            continue
        text = str(raw).upper()
        for allowed in INTENT_LABELS:
            if allowed in text:
                return allowed
    return ""


def _dict_get_case_insensitive(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    lowered = key.lower()
    for current_key, value in payload.items():
        if str(current_key).lower() == lowered:
            return value
    return None


def extract_axes(
    record: dict[str, Any], axis_names: list[str], field_candidates: list[str]
) -> tuple[dict[str, float], list[str]]:
    axis_blob: dict[str, Any] = {}

    for field in field_candidates:
        candidate = record.get(field)
        if isinstance(candidate, dict):
            axis_blob = candidate
            break

    if not axis_blob:
        labels = record.get("labels")
        if isinstance(labels, dict):
            for field in field_candidates:
                candidate = labels.get(field)
                if isinstance(candidate, dict):
                    axis_blob = candidate
                    break

    axes: dict[str, float] = {}
    present: list[str] = []
    for axis in axis_names:
        raw_value = None
        if axis_blob:
            raw_value = _dict_get_case_insensitive(axis_blob, axis)
        if raw_value is None:
            raw_value = _dict_get_case_insensitive(record, axis)
        if raw_value is None:
            axes[axis] = 0.0
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0
        else:
            present.append(axis)
        axes[axis] = value

    return axes, present


def extract_leaked_groups_mask(record: dict[str, Any]) -> int:
    direct = record.get("leakedGroupsMask")
    if direct is not None:
        return int_or_zero(direct)

    leak = record.get("leak")
    if isinstance(leak, dict):
        value = leak.get("groupsMask")
        if value is not None:
            return int_or_zero(value)

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("leakedGroupsMask")
        if value is not None:
            return int_or_zero(value)

    return 0


def has_harness_violation(record: dict[str, Any]) -> bool:
    direct_violation = record.get("harnessViolations")
    if normalize_boolish(direct_violation):
        return True

    illegal_attempts = record.get("illegalTransitionAttempts")
    if int_or_zero(illegal_attempts) > 0:
        return True

    harness = record.get("harness")
    if isinstance(harness, dict):
        if normalize_boolish(harness.get("violations")):
            return True
        if int_or_zero(harness.get("illegalTransitionAttempts")) > 0:
            return True

    return False


def collect_rejected_sessions(transcript_jsonl: Path) -> tuple[set[str], int]:
    rejected: set[str] = set()
    lines = 0
    for record in iter_jsonl(transcript_jsonl):
        lines += 1
        session_id = extract_session_id(record)
        if not session_id:
            continue
        if has_harness_violation(record):
            rejected.add(session_id)
    return rejected, lines


def build_records(
    transcript_jsonl: Path,
    axis_names: list[str],
    field_candidates: list[str],
    rejected_sessions: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    intent_records: list[dict[str, Any]] = []
    axis_records: list[dict[str, Any]] = []

    skipped_unknown_label = 0
    skipped_missing_session = 0
    skipped_axis_no_features = 0
    skipped_axis_leak = 0
    skipped_rejected_session = 0

    for line_no, record in enumerate(iter_jsonl(transcript_jsonl), 1):
        session_id = extract_session_id(record)
        if not session_id:
            skipped_missing_session += 1
            continue
        if session_id in rejected_sessions:
            skipped_rejected_session += 1
            continue

        label = extract_intent_label(record)
        if label not in INTENT_LABELS:
            skipped_unknown_label += 1
            continue

        sequence = record.get("sequenceId")
        sample_id = f"{session_id}:{sequence if sequence is not None else line_no}:{line_no}"
        text = extract_text(record)

        intent_records.append(
            {
                "task": "intent_router",
                "sampleId": sample_id,
                "sessionId": session_id,
                "text": text,
                "label": label,
                "timestamp": record.get("timestamp"),
                "sourceLine": line_no,
            }
        )

        if label != "VALID":
            continue

        leaked_groups_mask = extract_leaked_groups_mask(record)
        if leaked_groups_mask != 0:
            skipped_axis_leak += 1
            continue

        axes, present_axes = extract_axes(record, axis_names, field_candidates)
        if not present_axes:
            skipped_axis_no_features += 1
            continue

        axis_records.append(
            {
                "task": "axis_proposer",
                "sampleId": sample_id,
                "sessionId": session_id,
                "text": text,
                "label": "VALID",
                "axes": axes,
                "axesPresent": present_axes,
                "leakedGroupsMask": leaked_groups_mask,
                "timestamp": record.get("timestamp"),
                "sourceLine": line_no,
            }
        )

    summary = {
        "generatedAt": utc_now_iso(),
        "counts": {
            "intent_router": len(intent_records),
            "axis_proposer": len(axis_records),
            "rejected_sessions": len(rejected_sessions),
            "skipped_missing_session": skipped_missing_session,
            "skipped_unknown_label": skipped_unknown_label,
            "skipped_axis_no_features": skipped_axis_no_features,
            "skipped_axis_leak": skipped_axis_leak,
            "skipped_rejected_session_records": skipped_rejected_session,
        },
        "axes": axis_names,
        "intent_labels": list(INTENT_LABELS),
    }
    return intent_records, axis_records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training records from transcript JSONL.")
    parser.add_argument(
        "--transcript-jsonl",
        required=True,
        type=Path,
        help="Path to transcript JSONL in the external SSD dataset vault.",
    )
    parser.add_argument(
        "--axes-config",
        type=Path,
        default=Path("config/axes.yaml"),
        help="Axes configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/records"),
        help="Directory for generated task JSONL records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.transcript_jsonl.exists():
        raise FileNotFoundError(f"Input transcript JSONL not found: {args.transcript_jsonl}")

    axis_names, field_candidates = load_axes_config(args.axes_config)
    rejected_sessions, scanned_lines = collect_rejected_sessions(args.transcript_jsonl)
    intent_records, axis_records, summary = build_records(
        args.transcript_jsonl,
        axis_names,
        field_candidates,
        rejected_sessions,
    )
    summary["counts"]["lines_scanned"] = scanned_lines

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intent_path = args.output_dir / "intent_router_records.jsonl"
    axis_path = args.output_dir / "axis_proposer_records.jsonl"
    summary_path = args.output_dir / "build_summary.json"

    with intent_path.open("w", encoding="utf-8") as handle:
        for record in intent_records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
    with axis_path.open("w", encoding="utf-8") as handle:
        for record in axis_records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")

    write_json(summary_path, summary)

    print(f"Wrote intent records: {intent_path} ({len(intent_records)})")
    print(f"Wrote axis records: {axis_path} ({len(axis_records)})")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
