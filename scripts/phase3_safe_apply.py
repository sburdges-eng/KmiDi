#!/usr/bin/env python3
"""
Phase 3 safe-apply pipeline.

Reads reconciliation decisions and Phase 2 approvals, verifies the remote gate,
and produces a deterministic apply manifest. Optional --apply performs the
approved file copies only when the remote gate is clear.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ApplyRecord:
    batch: str
    status: str
    recovered_abs_path: str
    destination_path: str
    approval_state: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_fetch(workspace: Path) -> Tuple[bool, int, str]:
    proc = subprocess.run(
        ["git", "fetch", "origin", "--prune"],
        cwd=str(workspace),
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, proc.returncode, output.strip()


def read_decisions(path: Path) -> List[ApplyRecord]:
    records: List[ApplyRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            status = row.get("status", "")
            if status == "exact_match":
                records.append(
                    ApplyRecord(
                        batch="batch_a_exact",
                        status=status,
                        recovered_abs_path=row.get("recovered_abs_path", ""),
                        destination_path=row.get("destination_path", ""),
                        approval_state="auto_approved",
                    )
                )
            elif status == "duplicate_or_move":
                records.append(
                    ApplyRecord(
                        batch="batch_b_duplicate_or_move",
                        status=status,
                        recovered_abs_path=row.get("recovered_abs_path", ""),
                        destination_path=row.get("destination_path", ""),
                        approval_state="auto_approved",
                    )
                )
    return records


def read_phase2_candidates(path: Path) -> List[ApplyRecord]:
    records: List[ApplyRecord] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("phase2_decision") != "approved_auto_apply":
                continue
            records.append(
                ApplyRecord(
                    batch="batch_c_candidate_approved",
                    status=row.get("status", "candidate_match"),
                    recovered_abs_path=row.get("recovered_abs_path", ""),
                    destination_path=row.get("destination_path", ""),
                    approval_state="phase2_approved",
                )
            )
    return records


def read_phase2_conflicts(path: Path) -> List[ApplyRecord]:
    records: List[ApplyRecord] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("phase2_decision") != "approved_auto_apply":
                continue
            records.append(
                ApplyRecord(
                    batch="batch_d_conflict_approved",
                    status=row.get("status", "conflict"),
                    recovered_abs_path=row.get("recovered_abs_path", ""),
                    destination_path=row.get("top_candidate", ""),
                    approval_state="phase2_approved",
                )
            )
    return records


def dedupe(records: List[ApplyRecord]) -> List[ApplyRecord]:
    out: List[ApplyRecord] = []
    seen = set()
    for r in records:
        key = (r.batch, r.status, r.recovered_abs_path, r.destination_path, r.approval_state)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    out.sort(
        key=lambda r: (
            r.batch,
            r.destination_path,
            r.recovered_abs_path,
            r.status,
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    reports = workspace / "recovery_reports"
    reports.mkdir(parents=True, exist_ok=True)

    decisions_csv = reports / "decisions.csv"
    phase2_candidates_csv = reports / "phase2_candidates_auto_apply.csv"
    phase2_conflicts_csv = reports / "phase2_conflicts_auto_apply.csv"

    if not decisions_csv.exists():
        raise FileNotFoundError(f"missing required report: {decisions_csv}")

    remote_ok, return_code, fetch_output = run_fetch(workspace)
    remote_gate_path = reports / "phase3_remote_gate.json"
    remote_gate_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "remote_gate_passed": remote_ok,
                "command": ["git", "fetch", "origin", "--prune"],
                "return_code": return_code,
                "output": fetch_output,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    raw_records: List[ApplyRecord] = []
    raw_records.extend(read_decisions(decisions_csv))
    raw_records.extend(read_phase2_candidates(phase2_candidates_csv))
    raw_records.extend(read_phase2_conflicts(phase2_conflicts_csv))
    records = dedupe(raw_records)

    manifest_csv = reports / "phase3_apply_manifest.csv"
    summary_md = reports / "phase3_summary.md"
    git_batches_md = reports / "phase3_git_batches.md"

    counts: Dict[str, int] = {
        "batch_a_exact": 0,
        "batch_b_duplicate_or_move": 0,
        "batch_c_candidate_approved": 0,
        "batch_d_conflict_approved": 0,
    }
    outcomes: Dict[str, int] = {
        "verified_noop": 0,
        "copied": 0,
        "blocked_remote_gate": 0,
        "missing_source": 0,
        "missing_destination": 0,
        "missing_destination_parent": 0,
    }

    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch",
                "status",
                "approval_state",
                "recovered_abs_path",
                "destination_path",
                "source_exists",
                "destination_exists",
                "apply_result",
                "reason",
            ],
        )
        writer.writeheader()

        for rec in records:
            counts[rec.batch] += 1
            source = Path(rec.recovered_abs_path)
            source_exists = source.is_file()

            destination_exists = False
            destination = None
            if rec.destination_path:
                destination = workspace / rec.destination_path
                destination_exists = destination.exists()

            apply_result = "verified_noop"
            reason = "hash_equivalent_or_duplicate_mapping"

            if not source_exists:
                apply_result = "skipped"
                reason = "missing_source"
                outcomes["missing_source"] += 1
            elif not rec.destination_path:
                apply_result = "skipped"
                reason = "missing_destination"
                outcomes["missing_destination"] += 1
            elif rec.batch in {"batch_c_candidate_approved", "batch_d_conflict_approved"}:
                if not remote_ok:
                    apply_result = "blocked"
                    reason = "blocked_remote_gate"
                    outcomes["blocked_remote_gate"] += 1
                elif not args.apply:
                    apply_result = "pending"
                    reason = "dry_run_only"
                else:
                    parent = destination.parent if destination else None
                    if parent and not parent.exists():
                        parent.mkdir(parents=True, exist_ok=True)
                    if destination is None:
                        apply_result = "skipped"
                        reason = "missing_destination"
                        outcomes["missing_destination"] += 1
                    else:
                        shutil.copy2(str(source), str(destination))
                        apply_result = "applied"
                        reason = "approved_overwrite_or_add"
                        outcomes["copied"] += 1
            else:
                if destination and destination.exists():
                    # Exact/duplicate mappings are expected to be no-op.
                    try:
                        if source_exists and destination.is_file():
                            if sha256_file(source) == sha256_file(destination):
                                apply_result = "verified_noop"
                                reason = "source_equals_destination"
                                outcomes["verified_noop"] += 1
                            else:
                                apply_result = "blocked"
                                reason = "unexpected_hash_divergence"
                                outcomes["blocked_remote_gate"] += 1
                        else:
                            apply_result = "skipped"
                            reason = "destination_not_file"
                            outcomes["missing_destination"] += 1
                    except OSError:
                        apply_result = "skipped"
                        reason = "hash_check_failed"
                        outcomes["missing_destination"] += 1
                else:
                    if not remote_ok:
                        apply_result = "blocked"
                        reason = "blocked_remote_gate"
                        outcomes["blocked_remote_gate"] += 1
                    elif not args.apply:
                        apply_result = "pending"
                        reason = "dry_run_only"
                    else:
                        parent = destination.parent if destination else None
                        if destination is None:
                            apply_result = "skipped"
                            reason = "missing_destination"
                            outcomes["missing_destination"] += 1
                        elif parent and not parent.exists():
                            parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(source), str(destination))
                            apply_result = "applied"
                            reason = "restored_missing_canonical_file"
                            outcomes["copied"] += 1
                        else:
                            shutil.copy2(str(source), str(destination))
                            apply_result = "applied"
                            reason = "restored_missing_canonical_file"
                            outcomes["copied"] += 1

            writer.writerow(
                {
                    "batch": rec.batch,
                    "status": rec.status,
                    "approval_state": rec.approval_state,
                    "recovered_abs_path": rec.recovered_abs_path,
                    "destination_path": rec.destination_path,
                    "source_exists": str(source_exists).lower(),
                    "destination_exists": str(destination_exists).lower(),
                    "apply_result": apply_result,
                    "reason": reason,
                }
            )

    total_records = len(records)
    total_auto_candidates = counts["batch_a_exact"] + counts["batch_b_duplicate_or_move"]
    total_phase2_approved = counts["batch_c_candidate_approved"] + counts["batch_d_conflict_approved"]

    summary_md.write_text(
        "# Phase 3 Safe Apply Summary\n\n"
        f"- timestamp_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"- remote_gate_passed: {str(remote_ok).lower()}\n"
        f"- apply_mode: {'apply' if args.apply else 'dry_run'}\n"
        f"- total_manifest_records: {total_records}\n"
        f"- batch_a_exact: {counts['batch_a_exact']}\n"
        f"- batch_b_duplicate_or_move: {counts['batch_b_duplicate_or_move']}\n"
        f"- batch_c_candidate_approved: {counts['batch_c_candidate_approved']}\n"
        f"- batch_d_conflict_approved: {counts['batch_d_conflict_approved']}\n"
        f"- total_auto_class: {total_auto_candidates}\n"
        f"- total_phase2_approved: {total_phase2_approved}\n"
        f"- outcome_verified_noop: {outcomes['verified_noop']}\n"
        f"- outcome_copied: {outcomes['copied']}\n"
        f"- outcome_blocked_remote_gate: {outcomes['blocked_remote_gate']}\n"
        f"- outcome_missing_source: {outcomes['missing_source']}\n"
        f"- outcome_missing_destination: {outcomes['missing_destination']}\n",
        encoding="utf-8",
    )

    git_batches_md.write_text(
        "# Phase 3 Git Batching\n\n"
        "## Commit plan\n"
        "- Commit 1: exact path/hash safe imports (no-op verification or restores only)\n"
        "- Commit 2: deterministic duplicate/move mappings (no-op verification or restores only)\n"
        "- Commit 3: Phase 2 approved candidate/conflict resolutions\n\n"
        "## Gate\n"
        f"- remote gate passed: {str(remote_ok).lower()}\n"
        "- if gate is false, file content apply remains blocked and only reports are updated.\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "remote_gate_passed": remote_ok,
                "apply_mode": "apply" if args.apply else "dry_run",
                "manifest_records": total_records,
                "counts": counts,
                "outcomes": outcomes,
                "reports": {
                    "remote_gate": str(remote_gate_path),
                    "manifest": str(manifest_csv),
                    "summary": str(summary_md),
                    "git_batches": str(git_batches_md),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
