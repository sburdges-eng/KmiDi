#!/usr/bin/env python3
"""Validate packaged dataset manifests, checksums, and shard integrity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.training_common import (
    decode_jsonl_payload,
    load_json,
    sha256_file,
    validate_manifest_semantics,
)

SPLITS: tuple[str, ...] = ("train", "val", "test")


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def resolve_json_pointer(root_schema: dict[str, Any], pointer: str) -> dict[str, Any] | None:
    if not pointer.startswith("#/"):
        return None

    current: Any = root_schema
    for raw_part in pointer[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if not isinstance(current, dict):
        return None
    return current


def validate_against_schema(
    instance: Any,
    schema: dict[str, Any],
    path: str = "$",
    root_schema: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    root = root_schema or schema

    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = resolve_json_pointer(root, ref)
        if target is None:
            errors.append(f"{path}: unresolved $ref '{ref}'")
            return errors
        return validate_against_schema(instance, target, path=path, root_schema=root)

    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _json_type_matches(instance, expected_type):
        errors.append(f"{path}: expected type '{expected_type}'")
        return errors

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and instance not in enum_values:
        errors.append(f"{path}: value {instance!r} not in enum {enum_values!r}")

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: value {instance!r} does not match const {schema['const']!r}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in instance:
                    errors.append(f"{path}: missing required key '{key}'")

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in instance and isinstance(subschema, dict):
                    errors.extend(
                        validate_against_schema(instance[key], subschema, f"{path}.{key}", root)
                    )

        additional_allowed = schema.get("additionalProperties", True)
        if additional_allowed is False and isinstance(properties, dict):
            for key in instance.keys():
                if key not in properties:
                    errors.append(f"{path}: unexpected key '{key}'")

    if isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(instance):
                errors.extend(validate_against_schema(item, item_schema, f"{path}[{idx}]", root))

    return errors


def read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            checksum = parts[0]
            filename = parts[-1]
            checksums[filename] = checksum
    return checksums


def validate_shard_records(task_dir: Path, shard_name: str, expected_count: int) -> list[str]:
    errors: list[str] = []
    shard_path = task_dir / shard_name
    payload = shard_path.read_bytes()
    try:
        records = decode_jsonl_payload(payload)
    except Exception as exc:
        errors.append(f"{task_dir.name}/{shard_name}: decode failed ({exc})")
        return errors

    if len(records) != expected_count:
        errors.append(
            f"{task_dir.name}/{shard_name}: expected {expected_count} records, got {len(records)}"
        )
    return errors


def validate_task_package(task_dir: Path, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    manifest_path = task_dir / "manifest.json"
    checksums_path = task_dir / "checksums.txt"

    if not manifest_path.exists():
        return [f"{task_dir}: missing manifest.json"]
    if not checksums_path.exists():
        return [f"{task_dir}: missing checksums.txt"]

    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        return [f"{manifest_path}: manifest is not a JSON object"]

    errors.extend(validate_against_schema(manifest, schema))
    errors.extend(validate_manifest_semantics(manifest))

    checksums = read_checksums(checksums_path)
    if "manifest.json" not in checksums:
        errors.append(f"{checksums_path}: missing manifest.json checksum")

    manifest_hash = sha256_file(manifest_path)
    checksum_hash = checksums.get("manifest.json")
    if checksum_hash and checksum_hash != manifest_hash:
        errors.append(
            f"{manifest_path}: checksum mismatch (manifest={manifest_hash}, "
            f"checksums.txt={checksum_hash})"
        )

    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        errors.append(f"{manifest_path}: 'splits' must be object")
        return errors

    total_records = 0
    total_shards = 0
    for split_name in SPLITS:
        split_payload = splits.get(split_name, {})
        if not isinstance(split_payload, dict):
            errors.append(f"{manifest_path}: split '{split_name}' must be object")
            continue

        shards = split_payload.get("shards", [])
        if not isinstance(shards, list):
            errors.append(f"{manifest_path}: split '{split_name}' shards must be list")
            continue

        split_record_count = int(split_payload.get("recordCount", 0) or 0)
        counted_records = 0
        for shard in shards:
            if not isinstance(shard, dict):
                errors.append(f"{manifest_path}: split '{split_name}' shard must be object")
                continue
            filename = shard.get("file")
            if not isinstance(filename, str) or not filename:
                errors.append(f"{manifest_path}: split '{split_name}' shard missing filename")
                continue

            shard_path = task_dir / filename
            if not shard_path.exists():
                errors.append(f"{task_dir.name}/{filename}: missing shard file")
                continue

            expected_hash = shard.get("sha256")
            actual_hash = sha256_file(shard_path)
            if isinstance(expected_hash, str) and expected_hash and expected_hash != actual_hash:
                errors.append(f"{task_dir.name}/{filename}: sha256 mismatch")

            checksum_hash = checksums.get(filename)
            if checksum_hash and checksum_hash != actual_hash:
                errors.append(f"{checksums_path}: checksum mismatch for {filename}")
            if checksum_hash is None:
                errors.append(f"{checksums_path}: missing checksum entry for {filename}")

            record_count = int(shard.get("recordCount", 0) or 0)
            counted_records += record_count
            errors.extend(validate_shard_records(task_dir, filename, record_count))
            total_shards += 1

        if counted_records != split_record_count:
            errors.append(
                f"{manifest_path}: split '{split_name}' count mismatch "
                f"(manifest={split_record_count}, shards={counted_records})"
            )
        total_records += split_record_count

    totals = manifest.get("totals", {})
    if isinstance(totals, dict):
        manifest_total_records = int(totals.get("recordCount", 0) or 0)
        manifest_total_shards = int(totals.get("shardCount", 0) or 0)
        if manifest_total_records != total_records:
            errors.append(
                f"{manifest_path}: totals.recordCount mismatch "
                f"(manifest={manifest_total_records}, computed={total_records})"
            )
        if manifest_total_shards != total_shards:
            errors.append(
                f"{manifest_path}: totals.shardCount mismatch "
                f"(manifest={manifest_total_shards}, computed={total_shards})"
            )

    return errors


def discover_task_dirs(package_dir: Path) -> list[Path]:
    manifest_path = package_dir / "manifest.json"
    if manifest_path.exists():
        return [package_dir]

    task_dirs: list[Path] = []
    for child in sorted(package_dir.iterdir()):
        if child.is_dir() and (child / "manifest.json").exists():
            task_dirs.append(child)
    return task_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate packaged dataset shards and manifest integrity."
    )
    parser.add_argument(
        "--package-dir",
        required=True,
        type=Path,
        help="Path to PACKAGES/<package_id> or PACKAGES/<package_id>/<task>",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("config/dataset_manifest_schema.json"),
        help="Manifest JSON schema path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.package_dir.exists():
        raise FileNotFoundError(f"Package path not found: {args.package_dir}")
    if not args.schema.exists():
        raise FileNotFoundError(f"Schema not found: {args.schema}")

    schema = load_json(args.schema)
    if not isinstance(schema, dict):
        raise ValueError("Manifest schema must be a JSON object")

    task_dirs = discover_task_dirs(args.package_dir)
    if not task_dirs:
        raise FileNotFoundError(f"No task package directories found under: {args.package_dir}")

    all_errors: list[str] = []
    for task_dir in task_dirs:
        task_errors = validate_task_package(task_dir, schema)
        if task_errors:
            all_errors.extend(task_errors)
            print(f"{task_dir.name}: FAILED ({len(task_errors)} issue(s))")
        else:
            print(f"{task_dir.name}: OK")

    if all_errors:
        print("\nValidation failures:")
        for issue in all_errors:
            print(f"- {issue}")
        return 1

    print("Package validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
