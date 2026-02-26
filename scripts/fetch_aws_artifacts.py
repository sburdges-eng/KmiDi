#!/usr/bin/env python3
"""Fetch training artifacts from S3 (student-only by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.training_common import (
    DEFAULT_RUN_CONTRACT_PATH,
    build_s3_uri,
    load_run_contract,
    parse_s3_uri,
    run_contract_get,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch training artifacts from S3")
    parser.add_argument(
        "--s3-uri",
        default="",
        help="S3 URI to run root (s3://bucket/prefix/run-id) or output root when --run-id is provided",
    )
    parser.add_argument("--run-id", default="", help="Run ID to append to --s3-uri when needed")
    parser.add_argument(
        "--scope",
        default="student",
        choices=["student", "teacher", "all"],
        help="Artifact scope to fetch (default student-only).",
    )
    parser.add_argument(
        "--run-contract",
        type=Path,
        default=DEFAULT_RUN_CONTRACT_PATH,
        help="Run contract config path (YAML/JSON).",
    )
    parser.add_argument("--break-glass", action="store_true", help="Required for teacher artifact fetch")
    parser.add_argument("--break-glass-role-arn", default="", help="Break-glass role ARN for teacher fetch")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved S3/run details and exit")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/download"))
    parser.add_argument("--profile", default="")
    parser.add_argument("--region", default="")
    return parser.parse_args()


def validate_runs_prefix(uri: str) -> None:
    _, prefix = parse_s3_uri(uri)
    normalized = prefix.strip("/")
    if normalized != "training/runs" and not normalized.startswith("training/runs/"):
        raise ValueError(
            f"Invalid run artifact prefix: {uri}. Expected s3://<bucket>/training/runs[/<run-id>]."
        )


def key_is_local_safe(relative_key: str, scope: str = "student") -> bool:
    student_allowed = relative_key.startswith("artifacts/student/") or relative_key == "metrics.json"
    teacher_allowed = relative_key.startswith("artifacts/teacher/")
    if scope == "student":
        return student_allowed
    if scope == "teacher":
        return teacher_allowed
    return student_allowed or teacher_allowed


def load_run_contract_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_run_contract(path)


def resolve_fetch_uri(args: argparse.Namespace, contract: dict[str, Any]) -> str:
    base_uri = args.s3_uri.strip()
    if not base_uri:
        run_bucket = str(run_contract_get(contract, "s3", "runBucket", default="")).strip()
        run_prefix = str(run_contract_get(contract, "s3", "runPrefix", default="training/runs")).strip("/")
        if not run_bucket:
            raise ValueError("Missing run bucket: provide --s3-uri or set s3.runBucket in run contract")
        base_uri = build_s3_uri(run_bucket, run_prefix)
    full_uri = base_uri.rstrip("/")
    if args.run_id:
        full_uri = f"{full_uri}/{args.run_id}"
    return full_uri


def enforce_teacher_fetch_policy(args: argparse.Namespace, contract: dict[str, Any], default_profile: str) -> None:
    teacher_requested = args.scope in {"teacher", "all"}
    if not teacher_requested:
        if args.break_glass or args.break_glass_role_arn:
            raise ValueError("Break-glass flags are only valid with --scope teacher or --scope all")
        return

    if not args.break_glass:
        raise ValueError("Teacher artifact fetch requires --break-glass")
    if not args.profile:
        raise ValueError("Teacher artifact fetch requires explicit --profile")
    if default_profile and args.profile == default_profile:
        raise ValueError("Teacher artifact fetch requires a separate break-glass profile")
    if not args.break_glass_role_arn:
        raise ValueError("Teacher artifact fetch requires --break-glass-role-arn")

    required_profile = str(run_contract_get(contract, "security", "teacherFetch", "requiredProfile", default="")).strip()
    required_role = str(run_contract_get(contract, "security", "teacherFetch", "requiredRoleArn", default="")).strip()
    if required_profile and args.profile != required_profile:
        raise ValueError(
            f"Teacher artifact fetch requires profile '{required_profile}' from run contract; got '{args.profile}'"
        )
    if not required_role:
        raise ValueError(
            "security.teacherFetch.requiredRoleArn must be set in run contract for teacher fetch"
        )
    if args.break_glass_role_arn != required_role:
        raise ValueError(
            "Teacher artifact fetch role mismatch: provided role does not match security.teacherFetch.requiredRoleArn"
        )


def main() -> int:
    args = parse_args()
    contract = load_run_contract_optional(args.run_contract)
    default_profile = str(run_contract_get(contract, "aws", "profile", default="")).strip()
    default_region = str(run_contract_get(contract, "aws", "region", default="")).strip()

    args.profile = args.profile or default_profile
    args.region = args.region or default_region

    enforce_teacher_fetch_policy(args, contract, default_profile)

    full_uri = resolve_fetch_uri(args, contract)
    validate_runs_prefix(full_uri)
    bucket, prefix = parse_s3_uri(full_uri)
    prefix = prefix.strip("/")

    if args.dry_run:
        payload = {
            "mode": "dry-run",
            "runId": args.run_id,
            "resolved": {
                "bucket": bucket,
                "prefix": prefix,
                "uri": full_uri,
                "scope": args.scope,
                "profile": args.profile,
                "region": args.region,
                "breakGlass": args.break_glass,
                "breakGlassRoleArn": args.break_glass_role_arn,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required for artifact fetch") from exc

    session_kwargs: dict[str, Any] = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs)
    if args.scope in {"teacher", "all"}:
        sts_client = session.client("sts", region_name=args.region or None)
        role_session_name = f"fetch-teacher-{utc_now_iso().replace(':', '').replace('-', '')}"
        assumed = sts_client.assume_role(RoleArn=args.break_glass_role_arn, RoleSessionName=role_session_name)
        creds = assumed["Credentials"]
        role_session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=args.region or None,
        )
        s3_client = role_session.client("s3", region_name=args.region or None)
    else:
        s3_client = session.client("s3", region_name=args.region or None)

    paginator = s3_client.get_paginator("list_objects_v2")
    download_count = 0

    list_prefix = f"{prefix}/" if prefix else ""
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key:
                continue
            relative = key[len(list_prefix) :] if list_prefix else key
            if not key_is_local_safe(relative, args.scope):
                continue

            local_path = args.output_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, key, str(local_path))
            download_count += 1
            print(f"DOWNLOADED s3://{bucket}/{key} -> {local_path}")

    print(f"Fetch complete: downloaded={download_count} scope={args.scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
