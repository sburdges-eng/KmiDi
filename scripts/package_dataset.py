#!/usr/bin/env python3
"""Package deterministic split JSONL into shard archives and manifests."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.training_common import (
    DEFAULT_RUN_CONTRACT_PATH,
    build_s3_uri,
    chunked,
    compress_jsonl_records,
    iter_jsonl,
    load_run_contract,
    run_contract_get,
    sha256_file,
    shard_suffix_for_compression,
    utc_now_iso,
    validate_manifest_semantics,
    write_json,
)

TASKS: tuple[str, ...] = ("intent_router", "axis_proposer")
SPLITS: tuple[str, ...] = ("train", "val", "test")


def record_sort_key(record: dict[str, Any]) -> tuple[str, str, int]:
    session_id = str(record.get("sessionId", ""))
    sample_id = str(record.get("sampleId", ""))
    source_line = int(record.get("sourceLine", 0) or 0)
    return session_id, sample_id, source_line


def default_package_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"pkg-{stamp}"


def deterministic_timestamp(enabled: bool) -> str:
    if not enabled:
        return utc_now_iso()
    source_epoch = os.environ.get("SOURCE_DATE_EPOCH", "0").strip()
    try:
        epoch = int(source_epoch)
    except ValueError:
        epoch = 0
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def package_task(
    task: str,
    package_id: str,
    split_root: Path,
    package_dir: Path,
    records_per_shard: int,
    *,
    deterministic: bool,
) -> dict[str, Any]:
    task_input_dir = split_root / task
    task_output_dir = package_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schemaVersion": "1.0",
        "packageId": package_id,
        "task": task,
        "createdAt": deterministic_timestamp(deterministic),
        "hashAlgorithm": "sha256",
        "splits": {},
        "totals": {"recordCount": 0, "shardCount": 0},
    }

    all_compressions: set[str] = set()
    checksums: list[tuple[str, str]] = []

    for split_name in SPLITS:
        split_file = task_input_dir / f"{split_name}.jsonl"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split input file: {split_file}")
        records = list(iter_jsonl(split_file))
        records.sort(key=record_sort_key)

        split_info: dict[str, Any] = {"recordCount": len(records), "shards": []}
        for shard_index, chunk_records in enumerate(chunked(records, records_per_shard) if records else []):
            payload, compression = compress_jsonl_records(chunk_records)
            shard_suffix = shard_suffix_for_compression(compression)
            shard_name = f"{split_name}-{shard_index:04d}{shard_suffix}"
            shard_path = task_output_dir / shard_name
            shard_path.write_bytes(payload)

            checksum = sha256_file(shard_path)
            size_bytes = shard_path.stat().st_size
            all_compressions.add(compression)

            split_info["shards"].append(
                {
                    "file": shard_name,
                    "split": split_name,
                    "recordCount": len(chunk_records),
                    "sha256": checksum,
                    "sizeBytes": size_bytes,
                    "compression": compression,
                }
            )
            checksums.append((checksum, shard_name))

        manifest["totals"]["recordCount"] += len(records)
        manifest["totals"]["shardCount"] += len(split_info["shards"])
        manifest["splits"][split_name] = split_info

    if task == "intent_router":
        manifest["labelSpace"] = ["VALID", "ABSTAIN", "INVALID"]
    elif task == "axis_proposer":
        axis_names = []
        sample_path = task_input_dir / "train.jsonl"
        if sample_path.exists():
            for sample in iter_jsonl(sample_path):
                axes = sample.get("axes")
                if isinstance(axes, dict):
                    axis_names = sorted(str(name) for name in axes.keys())
                    break
        manifest["axisNames"] = axis_names

    if len(all_compressions) > 1:
        raise ValueError(
            f"{task}: mixed shard compression is not allowed in pre-training freeze mode: {sorted(all_compressions)}"
        )
    manifest["compression"] = next(iter(all_compressions)) if all_compressions else "identity"

    semantic_errors = validate_manifest_semantics(manifest)
    if semantic_errors:
        raise ValueError(f"{task}: invalid manifest semantics: {'; '.join(semantic_errors)}")

    manifest_path = task_output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest_checksum = sha256_file(manifest_path)
    checksums.append((manifest_checksum, "manifest.json"))

    checksums_path = task_output_dir / "checksums.txt"
    checksums.sort(key=lambda item: item[1])
    with checksums_path.open("w", encoding="utf-8") as handle:
        for checksum, filename in checksums:
            handle.write(f"{checksum}  {filename}\n")

    return {
        "task": task,
        "manifest": str(manifest_path),
        "checksums": str(checksums_path),
        "records": manifest["totals"]["recordCount"],
        "shards": manifest["totals"]["shardCount"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package training splits into shard archives.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("training/splits"),
        help="Split dataset root directory.",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("PACKAGES"),
        help="Output root for packaged datasets.",
    )
    parser.add_argument(
        "--package-id",
        default="",
        help="Package identifier. Defaults to UTC timestamp-based ID.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic metadata stamps (uses SOURCE_DATE_EPOCH, default 0).",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Pre-training freeze lane: requires explicit package ID and deterministic output.",
    )
    parser.add_argument(
        "--records-per-shard",
        type=int,
        default=5000,
        help="Maximum records per shard file.",
    )
    parser.add_argument(
        "--run-contract",
        type=Path,
        default=DEFAULT_RUN_CONTRACT_PATH,
        help="Run contract config path (YAML/JSON).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved package S3 destination details and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.records_per_shard <= 0:
        raise ValueError("--records-per-shard must be > 0")

    deterministic_mode = bool(args.deterministic or args.freeze)
    if args.freeze and not args.package_id:
        raise ValueError("--freeze requires --package-id (explicit package identity is mandatory).")
    package_id = args.package_id or default_package_id()
    contract: dict[str, Any] = {}
    if args.run_contract.exists():
        contract = load_run_contract(args.run_contract)

    package_bucket = str(run_contract_get(contract, "s3", "packageBucket", default="")).strip()
    package_prefix_root = str(run_contract_get(contract, "s3", "packagePrefix", default="training/packages")).strip("/")
    package_prefix = f"{package_prefix_root}/{package_id}" if package_prefix_root else package_id
    package_s3_uri = build_s3_uri(package_bucket, package_prefix) if package_bucket else ""

    if args.dry_run:
        payload = {
            "mode": "dry-run",
            "runId": "",
            "packageId": package_id,
            "packageDir": str(args.package_root / package_id),
            "deterministic": deterministic_mode,
            "freeze": bool(args.freeze),
            "resolved": {
                "packageBucket": package_bucket,
                "packagePrefix": package_prefix,
                "packageS3Uri": package_s3_uri,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    package_dir = args.package_root / package_id
    package_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for task in TASKS:
        result = package_task(
            task,
            package_id,
            args.split_dir,
            package_dir,
            args.records_per_shard,
            deterministic=deterministic_mode,
        )
        results.append(result)
        print(f"{task}: shards={result['shards']} records={result['records']}")

    summary = {
        "packageId": package_id,
        "createdAt": deterministic_timestamp(deterministic_mode),
        "deterministic": deterministic_mode,
        "freeze": bool(args.freeze),
        "tasks": results,
        "packageDir": str(package_dir),
    }
    summary_path = package_dir / "package_summary.json"
    write_json(summary_path, summary)
    print(f"Wrote package summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
