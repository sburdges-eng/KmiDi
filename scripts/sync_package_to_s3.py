#!/usr/bin/env python3
"""Incrementally sync packaged shards + manifests to S3 using SHA256 checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from scripts.training_common import (
    DEFAULT_RUN_CONTRACT_PATH,
    build_s3_uri,
    load_json,
    load_run_contract,
    parse_s3_uri,
    run_contract_get,
    sha256_bytes,
    sha256_file,
)


@dataclass(frozen=True)
class UploadTarget:
    local_path: Path
    key_suffix: str


@dataclass(frozen=True)
class SyncAction:
    local_path: Path
    s3_key: str
    local_sha256: str
    remote_sha256: str | None
    action: str  # upload | skip


def discover_task_dirs(package_dir: Path) -> list[Path]:
    task_dirs: list[Path] = []
    for child in sorted(package_dir.iterdir()):
        if child.is_dir() and (child / "manifest.json").exists():
            task_dirs.append(child)
    return task_dirs


def collect_upload_targets(package_dir: Path) -> list[UploadTarget]:
    targets: list[UploadTarget] = []
    seen: set[str] = set()

    for task_dir in discover_task_dirs(package_dir):
        task = task_dir.name
        manifest_path = task_dir / "manifest.json"
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid manifest: {manifest_path}")

        manifest_key = f"{task}/manifest.json"
        targets.append(UploadTarget(local_path=manifest_path, key_suffix=manifest_key))
        seen.add(manifest_key)

        splits = manifest.get("splits", {})
        if not isinstance(splits, dict):
            raise ValueError(f"Manifest missing splits object: {manifest_path}")

        for split_payload in splits.values():
            if not isinstance(split_payload, dict):
                continue
            shards = split_payload.get("shards", [])
            if not isinstance(shards, list):
                continue
            for shard in shards:
                if not isinstance(shard, dict):
                    continue
                filename = shard.get("file")
                if not isinstance(filename, str) or not filename:
                    continue
                key_suffix = f"{task}/{filename}"
                if key_suffix in seen:
                    continue
                local_path = task_dir / filename
                if not local_path.exists():
                    raise FileNotFoundError(f"Missing shard listed in manifest: {local_path}")
                targets.append(UploadTarget(local_path=local_path, key_suffix=key_suffix))
                seen.add(key_suffix)

    for task_dir in discover_task_dirs(package_dir):
        task = task_dir.name
        checksums_path = task_dir / "checksums.txt"
        if checksums_path.exists():
            key_suffix = f"{task}/checksums.txt"
            if key_suffix not in seen:
                targets.append(UploadTarget(local_path=checksums_path, key_suffix=key_suffix))
                seen.add(key_suffix)

    return sorted(targets, key=lambda item: item.key_suffix)


def plan_sync_actions(
    targets: Iterable[UploadTarget],
    key_prefix: str,
    remote_sha_lookup: Callable[[str], str | None],
) -> list[SyncAction]:
    actions: list[SyncAction] = []
    prefix = key_prefix.strip("/")

    for target in targets:
        key = f"{prefix}/{target.key_suffix}" if prefix else target.key_suffix
        local_sha = sha256_file(target.local_path)
        remote_sha = remote_sha_lookup(key)
        action = "skip" if remote_sha == local_sha else "upload"
        actions.append(
            SyncAction(
                local_path=target.local_path,
                s3_key=key,
                local_sha256=local_sha,
                remote_sha256=remote_sha,
                action=action,
            )
        )
    return actions


def build_remote_sha_lookup(s3_client: object, bucket: str) -> Callable[[str], str | None]:
    def _lookup(key: str) -> str | None:
        try:
            response = s3_client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            error = getattr(exc, "response", {}).get("Error", {})
            code = str(error.get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

        metadata = response.get("Metadata", {})
        if isinstance(metadata, dict):
            remote_sha = metadata.get("sha256") or metadata.get("x-amz-meta-sha256")
            if isinstance(remote_sha, str) and remote_sha:
                return remote_sha

        # Fallback for legacy objects without metadata: hash object bytes to compare safely.
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"]
        digest = sha256_bytes(body.read())
        return digest

    return _lookup


def upload_actions(
    s3_client: object,
    bucket: str,
    actions: list[SyncAction],
    dry_run: bool,
) -> tuple[int, int]:
    uploaded = 0
    skipped = 0

    for action in actions:
        if action.action == "skip":
            skipped += 1
            print(f"SKIP   s3://{bucket}/{action.s3_key}")
            continue

        if dry_run:
            print(f"DRYRUN upload {action.local_path} -> s3://{bucket}/{action.s3_key}")
            continue

        with action.local_path.open("rb") as handle:
            s3_client.put_object(
                Bucket=bucket,
                Key=action.s3_key,
                Body=handle,
                Metadata={"sha256": action.local_sha256},
            )
        uploaded += 1
        print(f"UPLOAD s3://{bucket}/{action.s3_key}")

    return uploaded, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally sync packaged shards/manifests to S3."
    )
    parser.add_argument(
        "--package-dir", required=True, type=Path, help="Path to PACKAGES/<package_id>"
    )
    parser.add_argument(
        "--s3-uri",
        default="",
        help="Destination S3 URI (s3://bucket/prefix). Defaults to run-contract package bucket/prefix + package id.",  # noqa: E501
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print resolved destination details and exit"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="No-op flag: sync is resumable by SHA256 and can be safely rerun.",
    )
    parser.add_argument(
        "--run-contract",
        type=Path,
        default=DEFAULT_RUN_CONTRACT_PATH,
        help="Run contract config path (YAML/JSON).",
    )
    parser.add_argument("--profile", default="", help="Optional AWS profile")
    parser.add_argument("--region", default="", help="Optional AWS region")
    return parser.parse_args()


def resolve_destination_s3_uri(
    args: argparse.Namespace, contract: dict[str, object], package_id: str
) -> str:
    explicit = args.s3_uri.strip()
    if explicit:
        return explicit

    bucket = str(run_contract_get(contract, "s3", "packageBucket", default="")).strip()
    prefix_root = str(
        run_contract_get(contract, "s3", "packagePrefix", default="training/packages")
    ).strip("/")
    if not bucket:
        raise ValueError(
            "Missing destination bucket: provide --s3-uri or set s3.packageBucket in run contract"
        )
    prefix = f"{prefix_root}/{package_id}" if prefix_root else package_id
    return build_s3_uri(bucket, prefix)


def main() -> int:
    args = parse_args()
    if not args.package_dir.exists():
        raise FileNotFoundError(f"Package directory not found: {args.package_dir}")

    contract: dict[str, object] = {}
    if args.run_contract.exists():
        contract = load_run_contract(args.run_contract)

    package_id = args.package_dir.name
    resolved_s3_uri = resolve_destination_s3_uri(args, contract, package_id)
    bucket, prefix = parse_s3_uri(resolved_s3_uri)
    resolved_profile = (
        args.profile or str(run_contract_get(contract, "aws", "profile", default="")).strip()
    )
    resolved_region = (
        args.region or str(run_contract_get(contract, "aws", "region", default="")).strip()
    )

    if args.dry_run:
        payload = {
            "mode": "dry-run",
            "runId": "",
            "packageId": package_id,
            "resolved": {
                "bucket": bucket,
                "prefix": prefix,
                "s3Uri": resolved_s3_uri,
                "profile": resolved_profile,
                "region": resolved_region,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    targets = collect_upload_targets(args.package_dir)
    if not targets:
        raise RuntimeError(f"No task manifests found under {args.package_dir}")

    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required for S3 sync") from exc

    session_kwargs = {}
    if resolved_profile:
        session_kwargs["profile_name"] = resolved_profile
    session = boto3.Session(**session_kwargs)
    s3_client = session.client("s3", region_name=resolved_region or None)

    lookup = build_remote_sha_lookup(s3_client, bucket)
    actions = plan_sync_actions(targets, prefix, lookup)

    uploads_needed = sum(1 for item in actions if item.action == "upload")
    skipped = len(actions) - uploads_needed
    print(f"Planned: total={len(actions)} upload={uploads_needed} skip={skipped}")

    uploaded, skipped_during = upload_actions(s3_client, bucket, actions, dry_run=False)
    print(f"Sync complete: uploaded={uploaded} skipped={skipped_during}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
