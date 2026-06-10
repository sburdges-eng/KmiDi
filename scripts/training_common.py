#!/usr/bin/env python3
"""Shared helpers for packaging, syncing, and AWS training scripts."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
INTENT_LABELS: tuple[str, ...] = ("VALID", "ABSTAIN", "INVALID")
DEFAULT_RUN_CONTRACT_PATH = Path("config/run_contract.yaml")
DATASET_SPLITS: tuple[str, ...] = ("train", "val", "test")
COMPRESSION_TO_SUFFIX: dict[str, str] = {"zstd": ".jsonl.zst", "identity": ".jsonl"}


def _expect_dict(payload: Any, path: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object")
    return payload


def _expect_non_empty_string(payload: Any, path: str) -> str:
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return payload.strip()


def _expect_bool(payload: Any, path: str) -> bool:
    if not isinstance(payload, bool):
        raise ValueError(f"{path} must be a boolean")
    return payload


def _expect_allowed_keys(
    payload: dict[str, Any],
    path: str,
    *,
    allowed: set[str],
    required: set[str] | None = None,
) -> dict[str, Any]:
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}")
    if required:
        missing = sorted(required - set(payload.keys()))
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")
    return payload


def _expect_positive_int(payload: Any, path: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int) or payload <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return payload


def _expect_positive_number(payload: Any, path: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)) or float(payload) <= 0:
        raise ValueError(f"{path} must be a positive number")
    return float(payload)


def _expect_non_negative_number(payload: Any, path: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)) or float(payload) < 0:
        raise ValueError(f"{path} must be a non-negative number")
    return float(payload)


def validate_run_contract(contract: dict[str, Any]) -> dict[str, Any]:
    contract = _expect_allowed_keys(
        contract,
        "run contract",
        allowed={"schemaVersion", "s3", "aws", "training", "security"},
        required={"schemaVersion", "s3", "aws", "training", "security"},
    )
    schema_version = _expect_non_empty_string(contract.get("schemaVersion"), "schemaVersion")
    if schema_version != "1.1":
        raise ValueError(f"schemaVersion must be '1.1' for freeze lanes, got '{schema_version}'")

    s3 = _expect_allowed_keys(
        _expect_dict(contract.get("s3"), "s3"),
        "s3",
        allowed={
            "packageBucket",
            "packagePrefix",
            "runBucket",
            "runPrefix",
            "teacherPrefix",
            "studentPrefix",
        },
        required={"packageBucket", "packagePrefix", "runBucket", "runPrefix"},
    )
    aws = _expect_allowed_keys(
        _expect_dict(contract.get("aws"), "aws"),
        "aws",
        allowed={"region", "profile", "breakGlassProfile"},
        required={"region", "profile", "breakGlassProfile"},
    )
    training = _expect_allowed_keys(
        _expect_dict(contract.get("training"), "training"),
        "training",
        allowed={
            "activePackageId",
            "defaultRunIdPrefix",
            "defaultInstanceType",
            "defaultMaxRuntimeHours",
            "budgetCapUsd",
            "checkpoints",
            "earlyStop",
            "ops",
        },
        required={
            "defaultRunIdPrefix",
            "defaultInstanceType",
            "defaultMaxRuntimeHours",
            "budgetCapUsd",
            "checkpoints",
            "earlyStop",
            "ops",
        },
    )
    security = _expect_allowed_keys(
        _expect_dict(contract.get("security"), "security"),
        "security",
        allowed={"teacherFetch"},
        required={"teacherFetch"},
    )

    _expect_non_empty_string(s3.get("packageBucket"), "s3.packageBucket")
    _expect_non_empty_string(s3.get("runBucket"), "s3.runBucket")
    _expect_non_empty_string(s3.get("packagePrefix"), "s3.packagePrefix")
    _expect_non_empty_string(s3.get("runPrefix"), "s3.runPrefix")
    if "teacherPrefix" in s3:
        _expect_non_empty_string(s3.get("teacherPrefix"), "s3.teacherPrefix")
    if "studentPrefix" in s3:
        _expect_non_empty_string(s3.get("studentPrefix"), "s3.studentPrefix")

    _expect_non_empty_string(aws.get("region"), "aws.region")
    if "profile" in aws:
        profile = aws.get("profile")
        if profile is not None and not isinstance(profile, str):
            raise ValueError("aws.profile must be a string")
    break_glass_profile = aws.get("breakGlassProfile")
    if break_glass_profile is not None and not isinstance(break_glass_profile, str):
        raise ValueError("aws.breakGlassProfile must be a string")

    if "activePackageId" in training:
        active_package_id = training.get("activePackageId")
        if active_package_id is not None and not isinstance(active_package_id, str):
            raise ValueError("training.activePackageId must be a string")
    _expect_non_empty_string(training.get("defaultRunIdPrefix"), "training.defaultRunIdPrefix")
    _expect_non_empty_string(training.get("defaultInstanceType"), "training.defaultInstanceType")
    _expect_positive_number(
        training.get("defaultMaxRuntimeHours"),
        "training.defaultMaxRuntimeHours",
    )
    budget_cap = _expect_positive_number(training.get("budgetCapUsd"), "training.budgetCapUsd")
    if budget_cap > 100.0:
        raise ValueError("training.budgetCapUsd must be <= 100.0")

    checkpoints = _expect_allowed_keys(
        _expect_dict(training.get("checkpoints"), "training.checkpoints"),
        "training.checkpoints",
        allowed={
            "defaultCadenceSteps",
            "minCadenceSteps",
            "maxCadenceSteps",
            "defaultMaxCheckpoints",
            "maxCheckpoints",
            "maxTotalCheckpoints",
        },
        required={
            "defaultCadenceSteps",
            "minCadenceSteps",
            "maxCadenceSteps",
            "defaultMaxCheckpoints",
            "maxCheckpoints",
            "maxTotalCheckpoints",
        },
    )
    default_cadence = _expect_positive_int(
        checkpoints.get("defaultCadenceSteps"), "training.checkpoints.defaultCadenceSteps"
    )
    min_cadence = _expect_positive_int(
        checkpoints.get("minCadenceSteps"), "training.checkpoints.minCadenceSteps"
    )
    max_cadence = _expect_positive_int(
        checkpoints.get("maxCadenceSteps"), "training.checkpoints.maxCadenceSteps"
    )
    if min_cadence > max_cadence:
        raise ValueError("training.checkpoints.minCadenceSteps must be <= maxCadenceSteps")
    if default_cadence < min_cadence or default_cadence > max_cadence:
        raise ValueError(
            "training.checkpoints.defaultCadenceSteps must be within "
            "[minCadenceSteps, maxCadenceSteps]"
        )
    default_keep = _expect_positive_int(
        checkpoints.get("defaultMaxCheckpoints"),
        "training.checkpoints.defaultMaxCheckpoints",
    )
    max_keep = _expect_positive_int(
        checkpoints.get("maxCheckpoints"), "training.checkpoints.maxCheckpoints"
    )
    max_total = _expect_positive_int(
        checkpoints.get("maxTotalCheckpoints"), "training.checkpoints.maxTotalCheckpoints"
    )
    if max_keep < default_keep:
        raise ValueError("training.checkpoints.maxCheckpoints must be >= defaultMaxCheckpoints")
    if max_total < max_keep + 1:
        raise ValueError(
            "training.checkpoints.maxTotalCheckpoints must allow "
            "keep-last-N + best policy (>= maxCheckpoints + 1)"
        )

    early_stop = _expect_allowed_keys(
        _expect_dict(training.get("earlyStop"), "training.earlyStop"),
        "training.earlyStop",
        allowed={
            "enabled",
            "metricPath",
            "patience",
            "minDelta",
            "minEpochs",
            "autoShutdownDefault",
        },
        required={
            "enabled",
            "metricPath",
            "patience",
            "minDelta",
            "minEpochs",
            "autoShutdownDefault",
        },
    )
    _expect_bool(early_stop.get("enabled"), "training.earlyStop.enabled")
    _expect_non_empty_string(early_stop.get("metricPath"), "training.earlyStop.metricPath")
    _expect_positive_int(early_stop.get("patience"), "training.earlyStop.patience")
    _expect_non_negative_number(early_stop.get("minDelta"), "training.earlyStop.minDelta")
    min_epochs = early_stop.get("minEpochs")
    if isinstance(min_epochs, bool) or not isinstance(min_epochs, int) or min_epochs < 0:
        raise ValueError("training.earlyStop.minEpochs must be a non-negative integer")
    _expect_bool(
        early_stop.get("autoShutdownDefault"),
        "training.earlyStop.autoShutdownDefault",
    )

    ops = _expect_allowed_keys(
        _expect_dict(training.get("ops"), "training.ops"),
        "training.ops",
        allowed={"preflightCommand", "tmuxSessionPrefix", "datasetCache"},
        required={"preflightCommand", "tmuxSessionPrefix", "datasetCache"},
    )
    _expect_non_empty_string(ops.get("preflightCommand"), "training.ops.preflightCommand")
    _expect_non_empty_string(ops.get("tmuxSessionPrefix"), "training.ops.tmuxSessionPrefix")
    dataset_cache = _expect_allowed_keys(
        _expect_dict(ops.get("datasetCache"), "training.ops.datasetCache"),
        "training.ops.datasetCache",
        allowed={
            "defaultMode",
            "ebsFallbackRoot",
            "allowStreamingFallback",
        },
        required={
            "defaultMode",
            "ebsFallbackRoot",
            "allowStreamingFallback",
        },
    )
    cache_mode = _expect_non_empty_string(
        dataset_cache.get("defaultMode"), "training.ops.datasetCache.defaultMode"
    )
    if cache_mode not in {"auto", "nvme", "ebs", "stream"}:
        raise ValueError(
            "training.ops.datasetCache.defaultMode must be one of " "auto|nvme|ebs|stream"
        )
    _expect_non_empty_string(
        dataset_cache.get("ebsFallbackRoot"),
        "training.ops.datasetCache.ebsFallbackRoot",
    )
    _expect_bool(
        dataset_cache.get("allowStreamingFallback"),
        "training.ops.datasetCache.allowStreamingFallback",
    )

    teacher_fetch = _expect_allowed_keys(
        _expect_dict(security.get("teacherFetch"), "security.teacherFetch"),
        "security.teacherFetch",
        allowed={"allowByDefault", "requiredProfile", "requiredRoleArn"},
        required={"allowByDefault", "requiredProfile", "requiredRoleArn"},
    )
    allow_by_default = _expect_bool(
        teacher_fetch.get("allowByDefault"),
        "security.teacherFetch.allowByDefault",
    )
    if allow_by_default:
        raise ValueError("security.teacherFetch.allowByDefault must be false (fail-closed)")
    required_profile = _expect_non_empty_string(
        teacher_fetch.get("requiredProfile"),
        "security.teacherFetch.requiredProfile",
    )
    required_role = teacher_fetch.get("requiredRoleArn")
    if required_role is not None and not isinstance(required_role, str):
        raise ValueError("security.teacherFetch.requiredRoleArn must be a string")
    if break_glass_profile:
        if break_glass_profile != required_profile:
            raise ValueError(
                "aws.breakGlassProfile must match " "security.teacherFetch.requiredProfile"
            )

    return contract


def shard_suffix_for_compression(compression: str) -> str:
    suffix = COMPRESSION_TO_SUFFIX.get(compression)
    if suffix is None:
        raise ValueError(f"Unsupported shard compression '{compression}'")
    return suffix


def _manifest_error(path: str, message: str, errors: list[str]) -> None:
    errors.append(f"{path}: {message}")


def validate_manifest_semantics(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    splits = manifest.get("splits")
    totals = manifest.get("totals")
    compression = manifest.get("compression")
    package_id = manifest.get("packageId")
    created_at = manifest.get("createdAt")

    if not isinstance(package_id, str) or not package_id.strip():
        _manifest_error("packageId", "must be a non-empty string", errors)
    if not isinstance(created_at, str) or not created_at.strip():
        _manifest_error("createdAt", "must be a non-empty timestamp", errors)

    if compression not in COMPRESSION_TO_SUFFIX:
        _manifest_error("compression", "must be one of zstd|identity", errors)

    if not isinstance(splits, dict):
        _manifest_error("splits", "must be an object", errors)
        return errors
    if not isinstance(totals, dict):
        _manifest_error("totals", "must be an object", errors)
        return errors

    aggregate_records = 0
    aggregate_shards = 0
    seen_compressions: set[str] = set()
    for split_name in DATASET_SPLITS:
        split_payload = splits.get(split_name)
        if not isinstance(split_payload, dict):
            _manifest_error(f"splits.{split_name}", "must be an object", errors)
            continue
        record_count = split_payload.get("recordCount")
        if not isinstance(record_count, int) or record_count < 0:
            _manifest_error(
                f"splits.{split_name}.recordCount",
                "must be a non-negative integer",
                errors,
            )
            continue
        shards = split_payload.get("shards")
        if not isinstance(shards, list):
            _manifest_error(f"splits.{split_name}.shards", "must be a list", errors)
            continue

        split_sum = 0
        for idx, shard in enumerate(shards):
            if not isinstance(shard, dict):
                _manifest_error(
                    f"splits.{split_name}.shards[{idx}]",
                    "must be an object",
                    errors,
                )
                continue
            shard_file = shard.get("file")
            shard_split = shard.get("split")
            shard_count = shard.get("recordCount")
            shard_compression = shard.get("compression")
            if not isinstance(shard_file, str) or not shard_file:
                _manifest_error(
                    f"splits.{split_name}.shards[{idx}].file",
                    "must be a non-empty string",
                    errors,
                )
                continue
            if shard_split != split_name:
                _manifest_error(
                    f"splits.{split_name}.shards[{idx}].split",
                    f"must match split '{split_name}'",
                    errors,
                )
            if not isinstance(shard_count, int) or shard_count < 0:
                _manifest_error(
                    f"splits.{split_name}.shards[{idx}].recordCount",
                    "must be a non-negative integer",
                    errors,
                )
            else:
                split_sum += shard_count
            if shard_compression not in COMPRESSION_TO_SUFFIX:
                _manifest_error(
                    f"splits.{split_name}.shards[{idx}].compression",
                    "must be one of zstd|identity",
                    errors,
                )
            else:
                seen_compressions.add(shard_compression)
                expected_suffix = shard_suffix_for_compression(shard_compression)
                if not shard_file.endswith(expected_suffix):
                    _manifest_error(
                        f"splits.{split_name}.shards[{idx}].file",
                        f"must end with '{expected_suffix}' for "
                        f"compression '{shard_compression}'",
                        errors,
                    )
        if split_sum != record_count:
            _manifest_error(
                f"splits.{split_name}.recordCount",
                f"must equal shard record sum ({split_sum})",
                errors,
            )
        aggregate_records += record_count
        aggregate_shards += len(shards)

    total_record_count = totals.get("recordCount")
    total_shard_count = totals.get("shardCount")
    if total_record_count != aggregate_records:
        _manifest_error(
            "totals.recordCount",
            f"must equal split record sum ({aggregate_records})",
            errors,
        )
    if total_shard_count != aggregate_shards:
        _manifest_error(
            "totals.shardCount",
            f"must equal split shard sum ({aggregate_shards})",
            errors,
        )
    if isinstance(compression, str):
        if len(seen_compressions) > 1:
            _manifest_error("compression", "must not be mixed across shards", errors)
        elif seen_compressions:
            expected_compression = next(iter(seen_compressions))
            if compression != expected_compression:
                _manifest_error(
                    "compression",
                    f"must match shard compression '{expected_compression}'",
                    errors,
                )

    return errors


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"Unable to parse {path}: install PyYAML or use JSON-compatible YAML"
            ) from exc
        return yaml.safe_load(text)


def load_run_contract(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Run contract not found: {path}")
    payload = load_yaml_or_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Run contract must be a mapping object: {path}")
    return validate_run_contract(payload)


def run_contract_get(contract: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = contract
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def resolve_checkpoint_settings(
    contract: dict[str, Any],
    cadence_steps: int,
    max_checkpoints: int,
) -> dict[str, int]:
    policy = run_contract_get(contract, "training", "checkpoints", default={})
    if not isinstance(policy, dict):
        policy = {}

    default_cadence = int(policy.get("defaultCadenceSteps", 10) or 10)
    min_cadence = int(policy.get("minCadenceSteps", 1) or 1)
    max_cadence = int(policy.get("maxCadenceSteps", 1000000) or 1000000)

    default_max_checkpoints = int(policy.get("defaultMaxCheckpoints", 3) or 3)
    max_checkpoints_limit = int(policy.get("maxCheckpoints", 20) or 20)
    max_total_checkpoints = int(
        policy.get("maxTotalCheckpoints", max_checkpoints_limit + 1) or (max_checkpoints_limit + 1)
    )

    resolved_cadence = cadence_steps if cadence_steps > 0 else default_cadence
    resolved_max_checkpoints = max_checkpoints if max_checkpoints > 0 else default_max_checkpoints

    if resolved_cadence < min_cadence or resolved_cadence > max_cadence:
        raise ValueError(
            f"checkpoint cadence {resolved_cadence} outside allowed range "
            f"[{min_cadence}, {max_cadence}] from run contract"
        )
    if resolved_max_checkpoints <= 0:
        raise ValueError("max checkpoints must be > 0")
    if resolved_max_checkpoints > max_checkpoints_limit:
        raise ValueError(
            f"max checkpoints {resolved_max_checkpoints} exceeds "
            f"run-contract limit {max_checkpoints_limit}"
        )
    if max_total_checkpoints <= 0:
        raise ValueError("max total checkpoints must be > 0")
    if max_total_checkpoints < resolved_max_checkpoints + 1:
        raise ValueError(
            f"max total checkpoints {max_total_checkpoints} is too small "
            f"for keep-last-N + best policy ({resolved_max_checkpoints + 1})"
        )

    return {
        "cadenceSteps": resolved_cadence,
        "maxCheckpoints": resolved_max_checkpoints,
        "minCadenceSteps": min_cadence,
        "maxCadenceSteps": max_cadence,
        "maxCheckpointsLimit": max_checkpoints_limit,
        "maxTotalCheckpoints": max_total_checkpoints,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def dumps_jsonl(records: Iterable[dict[str, Any]]) -> bytes:
    out: list[str] = []
    for record in records:
        out.append(json.dumps(record, separators=(",", ":"), sort_keys=True))
    return ("\n".join(out) + ("\n" if out else "")).encode("utf-8")


def stable_fraction(session_id: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{session_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def choose_split(session_id: str, seed: str, ratios: dict[str, float] | None = None) -> str:
    active = ratios or DEFAULT_SPLIT_RATIOS
    train_ratio = float(active.get("train", 0.8))
    val_ratio = float(active.get("val", 0.1))
    test_ratio = float(active.get("test", 0.1))
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to > 0")

    point = stable_fraction(session_id, seed)
    train_cut = train_ratio / ratio_sum
    val_cut = train_cut + (val_ratio / ratio_sum)
    if point < train_cut:
        return "train"
    if point < val_cut:
        return "val"
    return "test"


def chunked(records: list[dict[str, Any]], chunk_size: int) -> Iterator[list[dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    for idx in range(0, len(records), chunk_size):
        yield records[idx : idx + chunk_size]


def shard_indices_for_rank(rank: int, world_size: int, num_shards: int) -> list[int]:
    """Shard indices assigned to this rank (round-robin). For distributed cloud training."""
    if world_size <= 0 or num_shards <= 0:
        return list(range(num_shards)) if world_size <= 1 else []
    if rank < 0 or rank >= world_size:
        return []
    return [i for i in range(num_shards) if i % world_size == rank]


def filter_shards_for_rank(
    shards: list[dict[str, Any]], rank: int, world_size: int, train_only: bool = True
) -> list[dict[str, Any]]:
    """Return shard entries for this rank by position index. If not train_only,
    returns all (for val/test).
    """
    if not shards or world_size <= 1 or rank < 0:
        return shards
    if train_only:
        return [s for i, s in enumerate(shards) if i % world_size == rank]
    return shards


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def build_s3_uri(bucket: str, prefix: str = "") -> str:
    normalized_bucket = bucket.strip()
    if not normalized_bucket:
        raise ValueError("S3 bucket cannot be empty")
    normalized_prefix = prefix.strip("/")
    if normalized_prefix:
        return f"s3://{normalized_bucket}/{normalized_prefix}"
    return f"s3://{normalized_bucket}"


def normalize_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "none", "null", "no"}:
            return False
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)


def int_or_zero(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    return 0


def _compress_with_zstandard(payload: bytes) -> bytes | None:
    try:
        import zstandard  # type: ignore
    except Exception:
        return None
    compressor = zstandard.ZstdCompressor(level=3)
    return compressor.compress(payload)


def _decompress_with_zstandard(payload: bytes) -> bytes | None:
    try:
        import zstandard  # type: ignore
    except Exception:
        return None
    decompressor = zstandard.ZstdDecompressor()
    try:
        return decompressor.decompress(payload)
    except Exception:
        return None


def _compress_with_cli(payload: bytes) -> bytes | None:
    if shutil.which("zstd") is None:
        return None
    process = subprocess.run(
        ["zstd", "-q", "-c"],
        input=payload,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        return None
    return process.stdout


def _decompress_with_cli(payload: bytes) -> bytes | None:
    if shutil.which("zstd") is None:
        return None
    process = subprocess.run(
        ["zstd", "-q", "-d", "-c"],
        input=payload,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        return None
    return process.stdout


def compress_jsonl_records(records: Iterable[dict[str, Any]]) -> tuple[bytes, str]:
    raw_payload = dumps_jsonl(records)
    compressed = _compress_with_zstandard(raw_payload)
    if compressed is not None:
        return compressed, "zstd"
    compressed = _compress_with_cli(raw_payload)
    if compressed is not None:
        return compressed, "zstd"
    return raw_payload, "identity"


def decode_jsonl_payload(payload: bytes) -> list[dict[str, Any]]:
    decoded = _decompress_with_zstandard(payload)
    if decoded is None:
        decoded = _decompress_with_cli(payload)
    if decoded is None:
        decoded = payload
    text = decoded.decode("utf-8")
    records: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            records.append(obj)
    return records


def read_jsonl_or_zst(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".zst":
        return decode_jsonl_payload(path.read_bytes())
    return list(iter_jsonl(path))
