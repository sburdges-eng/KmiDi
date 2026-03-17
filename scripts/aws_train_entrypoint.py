#!/usr/bin/env python3
"""AWS GPU training entrypoint for IntentRouter and AxisProposer tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scripts.license_gate import (
    ALLOWED_LICENSES,
    ThirdPartyNotice,
    download_hf_model_snapshot,
    generate_third_party_notices,
    validate_hf_model_license,
)
from scripts.training_common import (
    DEFAULT_RUN_CONTRACT_PATH,
    build_s3_uri,
    filter_shards_for_rank,
    load_json,
    load_run_contract,
    normalize_boolish,
    parse_s3_uri,
    read_jsonl_or_zst,
    resolve_checkpoint_settings,
    run_contract_get,
    utc_now_iso,
    write_json,
)

TASKS: tuple[str, ...] = ("intent_router", "axis_proposer")
SPLITS: tuple[str, ...] = ("train", "val", "test")
INTENT_LABEL_TO_ID: dict[str, int] = {"VALID": 0, "ABSTAIN": 1, "INVALID": 2}
ID_TO_INTENT_LABEL: dict[int, str] = {value: key for key, value in INTENT_LABEL_TO_ID.items()}


@dataclass
class IntentDataset:
    texts: list[str]
    labels: list[int]


@dataclass
class AxisDataset:
    texts: list[str]
    targets: list[list[float]]
    axis_names: list[str]


def require_aws_gpu() -> None:
    aws_markers = [
        os.environ.get("AWS_EXECUTION_ENV"),
        os.environ.get("ECS_CONTAINER_METADATA_URI_V4"),
        os.environ.get("EC2_INSTANCE_ID"),
    ]
    is_aws = any(bool(marker) for marker in aws_markers)
    uuid_path = Path("/sys/hypervisor/uuid")
    if not is_aws and uuid_path.exists():
        uuid_value = uuid_path.read_text(encoding="utf-8").strip().lower()
        is_aws = uuid_value.startswith("ec2")

    if not is_aws:
        raise RuntimeError(
            "AWS check failed: training entrypoint can run only on AWS GPU instances."
        )

    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "GPU check failed: nvidia-smi not found. This runner must execute on AWS GPU instances."
        )
    process = subprocess.run(
        ["nvidia-smi", "-L"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if process.returncode != 0 or not process.stdout.strip():
        raise RuntimeError(
            "GPU check failed: no NVIDIA GPU detected. Training is restricted to AWS GPU instances."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AWS training entrypoint for listening-contract models."
    )
    parser.add_argument(
        "--package-s3-uri",
        default="",
        help="S3 URI where packaged task manifests/shards live. Defaults from run contract.",
    )
    parser.add_argument(
        "--package-local-dir",
        type=Path,
        default=None,
        help="Optional local package directory prepared by pre-cache.",
    )
    parser.add_argument(
        "--output-s3-uri",
        default="",
        help="S3 URI root for training artifacts. Defaults from run contract.",
    )
    parser.add_argument(
        "--package-id", default="", help="Package ID used when --package-s3-uri is omitted"
    )
    parser.add_argument("--run-id", default="", help="Run ID; default uses UTC timestamp")
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/listening-train"))
    parser.add_argument("--teacher-hf-model", default="", help="Optional HF teacher model ID")
    parser.add_argument("--teacher-hf-revision", default="main")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--allow-non-aws", action="store_true", help="Bypass AWS GPU enforcement (debug only)"
    )
    parser.add_argument(
        "--run-contract",
        type=Path,
        default=DEFAULT_RUN_CONTRACT_PATH,
        help="Run contract config path (YAML/JSON).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print resolved S3/run details and exit"
    )
    parser.add_argument(
        "--checkpoint-cadence-steps",
        type=int,
        default=0,
        help="Checkpoint cadence in training steps/epochs (0 uses run-contract default).",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=0,
        help="Max rolling checkpoints to keep per model family (0 uses run-contract default).",
    )
    parser.add_argument(
        "--max-total-checkpoints",
        type=int,
        default=0,
        help="Absolute cap for checkpoint files after pruning (0 uses run-contract default).",
    )
    parser.add_argument("--early-stop-enabled", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=-1.0)
    parser.add_argument("--early-stop-min-epochs", type=int, default=0)
    parser.add_argument("--early-stop-metric", default="", help="Dot path metric in eval summary")
    parser.add_argument(
        "--auto-shutdown-on-complete",
        action="store_true",
        help="Shutdown instance after early-stop completion",
    )
    parser.add_argument("--intent-dim", type=int, default=2048)
    parser.add_argument("--student-intent-dim", type=int, default=512)
    parser.add_argument("--intent-epochs", type=int, default=40)
    parser.add_argument("--student-intent-epochs", type=int, default=50)
    parser.add_argument("--intent-lr", type=float, default=0.5)
    parser.add_argument("--distill-alpha", type=float, default=0.7, help="Weight on hard labels")
    parser.add_argument("--axis-l2", type=float, default=1.0)
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Distributed rank (0..world_size-1). Default: RANK env or 0.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of distributed workers. Default: WORLD_SIZE env or 1.",
    )
    args = parser.parse_args()
    if args.rank is None:
        args.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if args.world_size is None:
        args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return args


def load_run_contract_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_run_contract(path)


def resolve_training_uris(
    args: argparse.Namespace,
    contract: dict[str, Any],
) -> tuple[str, str]:
    package_uri = args.package_s3_uri.strip()
    if not package_uri:
        package_id = (
            args.package_id.strip()
            or str(run_contract_get(contract, "training", "activePackageId", default="")).strip()
        )
        package_bucket = str(run_contract_get(contract, "s3", "packageBucket", default="")).strip()
        package_prefix_root = str(
            run_contract_get(contract, "s3", "packagePrefix", default="training/packages")
        ).strip("/")
        if not package_id:
            raise ValueError(
                "Missing package id: provide --package-s3-uri, --package-id, or "
                "training.activePackageId"
            )
        if not package_bucket:
            raise ValueError(
                "Missing package bucket: provide --package-s3-uri or set s3.packageBucket"
            )
        package_prefix = (
            f"{package_prefix_root}/{package_id}" if package_prefix_root else package_id
        )
        package_uri = build_s3_uri(package_bucket, package_prefix)

    output_uri = args.output_s3_uri.strip()
    if not output_uri:
        run_bucket = str(run_contract_get(contract, "s3", "runBucket", default="")).strip()
        run_prefix = str(
            run_contract_get(contract, "s3", "runPrefix", default="training/runs")
        ).strip("/")
        if not run_bucket:
            raise ValueError("Missing run bucket: provide --output-s3-uri or set s3.runBucket")
        output_uri = build_s3_uri(run_bucket, run_prefix)

    return package_uri, output_uri


def resolve_max_total_checkpoints(args: argparse.Namespace, contract: dict[str, Any]) -> int:
    policy = run_contract_get(contract, "training", "checkpoints", default={})
    if not isinstance(policy, dict):
        policy = {}

    resolved = (
        args.max_total_checkpoints
        if args.max_total_checkpoints > 0
        else int(
            policy.get("maxTotalCheckpoints", args.max_checkpoints + 1)
            or (args.max_checkpoints + 1)
        )
    )
    if resolved <= 0:
        raise ValueError("max total checkpoints must be > 0")
    if resolved < args.max_checkpoints + 1:
        raise ValueError(
            f"max total checkpoints {resolved} is too small for keep-last-N + best "
            f"policy ({args.max_checkpoints + 1})"
        )
    return resolved


def resolve_early_stop_settings(
    args: argparse.Namespace, contract: dict[str, Any]
) -> dict[str, Any]:
    policy = run_contract_get(contract, "training", "earlyStop", default={})
    if not isinstance(policy, dict):
        policy = {}

    enabled = args.early_stop_enabled or normalize_boolish(policy.get("enabled", False))
    patience = (
        args.early_stop_patience
        if args.early_stop_patience > 0
        else int(policy.get("patience", 0) or 0)
    )
    min_delta = (
        args.early_stop_min_delta
        if args.early_stop_min_delta >= 0
        else float(policy.get("minDelta", 0.0) or 0.0)
    )
    min_epochs = (
        args.early_stop_min_epochs
        if args.early_stop_min_epochs > 0
        else int(policy.get("minEpochs", 0) or 0)
    )
    metric_path = args.early_stop_metric.strip() or str(
        policy.get("metricPath", "student.intent.val_macro_f1")
    )
    auto_shutdown = bool(args.auto_shutdown_on_complete)

    if enabled and patience <= 0:
        raise ValueError("Early stop enabled but patience is not > 0")
    if min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0")

    return {
        "enabled": enabled,
        "patience": patience,
        "minDelta": min_delta,
        "minEpochs": min_epochs,
        "metricPath": metric_path,
        "autoShutdown": auto_shutdown,
    }


def _checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    return checkpoint_dir / f"checkpoint-step-{int(step):06d}.json"


def write_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model_payload: dict[str, Any],
    metric_name: str = "",
    metric_value: float | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "createdAt": utc_now_iso(),
        "step": int(step),
        "model": model_payload,
    }
    if metric_name:
        payload["metrics"] = {metric_name: metric_value}
    path = _checkpoint_path(checkpoint_dir, step)
    write_model_artifact(path, payload)
    return path


def prune_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int,
    metric_name: str = "",
    higher_is_better: bool = True,
    final_export_path: Path | None = None,
    max_total_checkpoints: int = 0,
) -> dict[str, Any]:
    if keep_last_n <= 0:
        raise ValueError("keep_last_n must be > 0")

    final_export_name = ""
    if final_export_path is not None:
        final_export_name = final_export_path.name
        if not final_export_path.exists():
            raise RuntimeError(
                f"Final export missing before checkpoint pruning: {final_export_path}"
            )

    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint-step-*.json"))
    if not checkpoint_files:
        return {"total": 0, "kept": 0, "pruned": 0, "best": "", "finalExport": final_export_name}

    def _step(path: Path) -> int:
        stem = path.stem
        suffix = stem.replace("checkpoint-step-", "", 1)
        try:
            return int(suffix)
        except ValueError:
            return -1

    ordered = sorted(checkpoint_files, key=_step)
    keep: set[Path] = set(ordered[-keep_last_n:])
    best_path: Path | None = None

    if metric_name:
        scored: list[tuple[float, Path]] = []
        for path in ordered:
            try:
                payload = load_json(path)
            except Exception:
                continue
            metrics = payload.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            raw_metric = metrics.get(metric_name)
            if raw_metric is None:
                continue
            try:
                metric = float(raw_metric)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(metric):
                continue
            scored.append((metric, path))
        if scored:
            best_metric, best_path = (
                max(scored, key=lambda item: item[0])
                if higher_is_better
                else min(scored, key=lambda item: item[0])
            )
            _ = best_metric
            keep.add(best_path)

    if max_total_checkpoints > 0 and len(keep) > max_total_checkpoints:
        kept_ordered = sorted(keep, key=_step)
        protected: set[Path] = set()
        if best_path is not None:
            protected.add(best_path)
        while len(kept_ordered) > max_total_checkpoints:
            removable = [path for path in kept_ordered if path not in protected]
            if not removable:
                break
            drop = removable[0]
            keep.discard(drop)
            kept_ordered.remove(drop)

    pruned = 0
    for path in ordered:
        if path in keep:
            continue
        path.unlink(missing_ok=True)
        pruned += 1

    return {
        "total": len(ordered),
        "kept": len(keep),
        "pruned": pruned,
        "best": best_path.name if best_path is not None else "",
        "finalExport": final_export_name,
    }


def assert_teacher_prefix(upload_root: Path) -> None:
    teacher_root = upload_root / "artifacts" / "teacher"
    if not teacher_root.exists():
        raise RuntimeError("Teacher artifact root missing")

    for path in upload_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(upload_root).as_posix()
        if "teacher" in relative and not relative.startswith("artifacts/teacher/"):
            raise RuntimeError(f"Teacher data must remain under artifacts/teacher/: {relative}")


def hash_token(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % dim


def vectorize_texts(texts: list[str], dim: int):
    import numpy as np

    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        if not text:
            continue
        tokens = text.lower().split()
        if not tokens:
            continue
        inv = 1.0 / float(len(tokens))
        for token in tokens:
            matrix[row, hash_token(token, dim)] += inv
    return matrix


def softmax(logits):
    import numpy as np

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=1, keepdims=True)
    return exp_values / denom


def one_hot(labels: list[int], num_classes: int):
    import numpy as np

    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    for idx, label in enumerate(labels):
        out[idx, label] = 1.0
    return out


def train_intent_softmax(
    x_train,
    y_train: list[int],
    epochs: int,
    lr: float,
    l2: float = 1e-4,
    checkpoint_cadence_steps: int = 0,
    checkpoint_callback: Callable[[int, Any, Any], None] | None = None,
    epoch_callback: Callable[[int, Any, Any], bool] | None = None,
):
    import numpy as np

    n_samples, dim = x_train.shape
    num_classes = len(INTENT_LABEL_TO_ID)
    weights = np.zeros((dim, num_classes), dtype=np.float32)
    bias = np.zeros((num_classes,), dtype=np.float32)
    targets = one_hot(y_train, num_classes)

    last_epoch = 0
    for epoch in range(1, epochs + 1):
        logits = x_train @ weights + bias
        probs = softmax(logits)
        grad_logits = (probs - targets) / max(1, n_samples)
        grad_w = x_train.T @ grad_logits + (l2 * weights)
        grad_b = grad_logits.sum(axis=0)
        weights -= lr * grad_w
        bias -= lr * grad_b
        last_epoch = epoch

        if (
            checkpoint_callback
            and checkpoint_cadence_steps > 0
            and (epoch % checkpoint_cadence_steps == 0)
        ):
            checkpoint_callback(epoch, weights.copy(), bias.copy())

        if epoch_callback and not epoch_callback(epoch, weights.copy(), bias.copy()):
            break

    return weights, bias, last_epoch


def train_intent_student(
    x_student,
    y_train: list[int],
    teacher_probs,
    epochs: int,
    lr: float,
    alpha: float,
    l2: float = 1e-4,
    checkpoint_cadence_steps: int = 0,
    checkpoint_callback: Callable[[int, Any, Any], None] | None = None,
    epoch_callback: Callable[[int, Any, Any], bool] | None = None,
):
    import numpy as np

    n_samples, dim = x_student.shape
    num_classes = len(INTENT_LABEL_TO_ID)
    weights = np.zeros((dim, num_classes), dtype=np.float32)
    bias = np.zeros((num_classes,), dtype=np.float32)

    hard_targets = one_hot(y_train, num_classes)
    mixed_targets = (alpha * hard_targets) + ((1.0 - alpha) * teacher_probs)

    last_epoch = 0
    for epoch in range(1, epochs + 1):
        logits = x_student @ weights + bias
        probs = softmax(logits)
        grad_logits = (probs - mixed_targets) / max(1, n_samples)
        grad_w = x_student.T @ grad_logits + (l2 * weights)
        grad_b = grad_logits.sum(axis=0)
        weights -= lr * grad_w
        bias -= lr * grad_b
        last_epoch = epoch

        if (
            checkpoint_callback
            and checkpoint_cadence_steps > 0
            and (epoch % checkpoint_cadence_steps == 0)
        ):
            checkpoint_callback(epoch, weights.copy(), bias.copy())

        if epoch_callback and not epoch_callback(epoch, weights.copy(), bias.copy()):
            break

    return weights, bias, last_epoch


def intent_predict_probs(x_data, weights, bias):
    logits = x_data @ weights + bias
    return softmax(logits)


def intent_predict_labels(x_data, weights, bias) -> list[int]:
    import numpy as np

    probs = intent_predict_probs(x_data, weights, bias)
    return [int(item) for item in np.argmax(probs, axis=1)]


def macro_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    scores: list[float] = []
    for label in range(num_classes):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        recall = tp / float(tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return sum(scores) / float(len(scores))


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / float(len(y_true))


def train_axis_ridge(x_train, y_train, l2: float):
    import numpy as np

    if x_train.size == 0:
        raise RuntimeError("AxisProposer train set is empty")

    n_samples, dim = x_train.shape
    _, out_dim = y_train.shape
    x_aug = np.concatenate([x_train, np.ones((n_samples, 1), dtype=np.float32)], axis=1)

    gram = x_aug.T @ x_aug
    gram += l2 * np.eye(dim + 1, dtype=np.float32)
    rhs = x_aug.T @ y_train
    coeff = np.linalg.solve(gram, rhs)
    weights = coeff[:-1, :]
    bias = coeff[-1, :]
    return weights, bias


def axis_predict(x_data, weights, bias):
    return (x_data @ weights) + bias


def mean_squared_error(y_true, y_pred) -> float:
    import numpy as np

    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_true - y_pred) ** 2))


def load_manifest_and_shards(
    s3_client: object,
    package_uri: str,
    local_root: Path,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, dict[str, Any]]:
    """Download manifest and shards. For distributed (world_size > 1), only train
    shards for this rank are downloaded.
    """
    bucket, prefix = parse_s3_uri(package_uri)
    manifests: dict[str, dict[str, Any]] = {}

    for task in TASKS:
        task_dir = local_root / task
        task_dir.mkdir(parents=True, exist_ok=True)

        manifest_key = f"{prefix}/{task}/manifest.json" if prefix else f"{task}/manifest.json"
        manifest_path = task_dir / "manifest.json"
        s3_client.download_file(bucket, manifest_key, str(manifest_path))
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid manifest object for task {task}")
        manifests[task] = manifest

        splits = manifest.get("splits", {})
        if not isinstance(splits, dict):
            raise ValueError(f"Manifest missing splits object for task {task}")

        for split_name, split_payload in splits.items():
            if not isinstance(split_payload, dict):
                continue
            shards = split_payload.get("shards", [])
            if not isinstance(shards, list):
                continue
            # For distributed: only this rank's train shards; val/test get all
            if split_name == "train" and world_size > 1:
                shards = filter_shards_for_rank(shards, rank, world_size, train_only=True)
            for shard in shards:
                if not isinstance(shard, dict):
                    continue
                shard_name = shard.get("file")
                if not isinstance(shard_name, str) or not shard_name:
                    continue
                shard_path = task_dir / shard_name
                shard_key = (
                    f"{prefix}/{task}/{shard_name}" if prefix else f"{task}/{shard_name}"
                )
                s3_client.download_file(bucket, shard_key, str(shard_path))

    return manifests


def load_manifest_from_local_package(package_root: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        task_dir = package_root / task
        manifest_path = task_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Local package missing manifest: {manifest_path}")
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid local manifest payload: {manifest_path}")
        manifests[task] = manifest

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
                shard_name = shard.get("file")
                if not isinstance(shard_name, str) or not shard_name:
                    continue
                shard_path = task_dir / shard_name
                if not shard_path.exists():
                    raise FileNotFoundError(
                        f"Local package shard missing: {shard_path}"
                    )

    return manifests


def load_task_split_records(
    task_dir: Path,
    manifest: dict[str, Any],
    split_name: str,
    rank: int = 0,
    world_size: int = 1,
) -> list[dict[str, Any]]:
    """Load records for a split. For train and world_size > 1, only this rank's
    shards are loaded.
    """
    split_payload = manifest.get("splits", {}).get(split_name, {})
    if not isinstance(split_payload, dict):
        return []

    shards = split_payload.get("shards", [])
    if not isinstance(shards, list):
        return []
    if split_name == "train" and world_size > 1:
        shards = filter_shards_for_rank(shards, rank, world_size, train_only=True)
    records: list[dict[str, Any]] = []
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        filename = shard.get("file")
        if not isinstance(filename, str) or not filename:
            continue
        shard_path = task_dir / filename
        records.extend(read_jsonl_or_zst(shard_path))
    return records


def build_intent_dataset(records: list[dict[str, Any]]) -> IntentDataset:
    texts: list[str] = []
    labels: list[int] = []
    for record in records:
        label = str(record.get("label", "")).upper()
        if label not in INTENT_LABEL_TO_ID:
            continue
        texts.append(str(record.get("text", "")))
        labels.append(INTENT_LABEL_TO_ID[label])
    return IntentDataset(texts=texts, labels=labels)


def build_axis_dataset(records: list[dict[str, Any]], axis_names: list[str]) -> AxisDataset:
    texts: list[str] = []
    targets: list[list[float]] = []

    for record in records:
        axes = record.get("axes")
        if not isinstance(axes, dict):
            continue
        vector: list[float] = []
        for axis_name in axis_names:
            raw = axes.get(axis_name, 0.0)
            try:
                vector.append(float(raw))
            except (TypeError, ValueError):
                vector.append(0.0)
        texts.append(str(record.get("text", "")))
        targets.append(vector)

    return AxisDataset(texts=texts, targets=targets, axis_names=axis_names)


def write_model_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_metric_from_summary(path: Path, metric_path: str) -> float | None:
    if not path.exists():
        return None
    payload = load_json(path)
    current: Any = payload
    for segment in metric_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    try:
        value = float(current)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def upload_directory_to_s3(s3_client: object, local_root: Path, s3_uri: str) -> int:
    bucket, prefix = parse_s3_uri(s3_uri)
    uploaded = 0

    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_root).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        s3_client.upload_file(str(path), bucket, key)
        uploaded += 1
    return uploaded


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required on AWS training hosts") from exc

    try:
        import numpy as np  # noqa: F401
    except Exception as exc:
        raise RuntimeError("numpy is required on AWS training hosts") from exc

    workdir = args.workdir
    package_local = (
        args.package_local_dir if args.package_local_dir is not None else workdir / "package"
    )
    package_local = package_local.resolve()
    upload_root = workdir / "upload_bundle"
    artifacts_root = upload_root / "artifacts"
    teacher_dir = artifacts_root / "teacher"
    student_dir = artifacts_root / "student"
    teacher_ckpt_dir = teacher_dir / "checkpoints"
    student_ckpt_dir = student_dir / "checkpoints"
    model_cache_dir = workdir / "models"

    package_local.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)
    student_dir.mkdir(parents=True, exist_ok=True)
    teacher_ckpt_dir.mkdir(parents=True, exist_ok=True)
    student_ckpt_dir.mkdir(parents=True, exist_ok=True)

    session = boto3.Session()
    s3_client = session.client("s3")

    if args.package_local_dir is not None:
        manifests = load_manifest_from_local_package(package_local)
    else:
        manifests = load_manifest_and_shards(
            s3_client=s3_client,
            package_uri=args.package_s3_uri,
            local_root=package_local,
            rank=args.rank,
            world_size=args.world_size,
        )

    notices: list[ThirdPartyNotice] = []
    hf_token = os.environ.get(args.hf_token_env, "")
    if args.teacher_hf_model:
        license_id, _ = validate_hf_model_license(
            args.teacher_hf_model, token=hf_token, allowlist=ALLOWED_LICENSES
        )
        notices.append(
            ThirdPartyNotice(
                name=args.teacher_hf_model,
                source=f"https://huggingface.co/{args.teacher_hf_model}",
                license_id=license_id,
            )
        )
        download_hf_model_snapshot(
            model_id=args.teacher_hf_model,
            destination_dir=model_cache_dir / "teacher_hf",
            token=hf_token,
            revision=args.teacher_hf_revision,
        )

    notices_path = upload_root / "THIRD_PARTY_NOTICES.md"
    generate_third_party_notices(notices, notices_path)

    intent_manifest = manifests["intent_router"]
    axis_manifest = manifests["axis_proposer"]

    intent_data = {
        split: build_intent_dataset(
            load_task_split_records(
                package_local / "intent_router",
                intent_manifest,
                split,
                rank=args.rank,
                world_size=args.world_size,
            )
        )
        for split in SPLITS
    }

    axis_names = axis_manifest.get("axisNames", [])
    if not isinstance(axis_names, list) or not axis_names:
        axis_names = ["valence", "energy", "density", "complexity"]
    axis_names = [str(item) for item in axis_names]

    axis_data = {
        split: build_axis_dataset(
            load_task_split_records(
                package_local / "axis_proposer",
                axis_manifest,
                split,
                rank=args.rank,
                world_size=args.world_size,
            ),
            axis_names,
        )
        for split in SPLITS
    }

    if len(intent_data["train"].labels) == 0:
        raise RuntimeError("IntentRouter training split is empty")
    if len(axis_data["train"].targets) == 0:
        raise RuntimeError("AxisProposer training split is empty")

    x_intent_train = vectorize_texts(intent_data["train"].texts, args.intent_dim)
    x_intent_val = vectorize_texts(intent_data["val"].texts, args.intent_dim)
    x_intent_test = vectorize_texts(intent_data["test"].texts, args.intent_dim)

    student_eval_summary_path = student_dir / "eval_summary.json"
    early_stop_state: dict[str, Any] = {
        "enabled": bool(args.early_stop_enabled),
        "metricPath": str(args.early_stop_metric),
        "patience": int(args.early_stop_patience),
        "minDelta": float(args.early_stop_min_delta),
        "minEpochs": int(args.early_stop_min_epochs),
        "bestMetric": None,
        "bestEpoch": 0,
        "stalledEpochs": 0,
        "stopped": False,
        "stopReason": "",
        "stopEpoch": 0,
    }
    best_student_weights: Any | None = None
    best_student_bias: Any | None = None

    def _teacher_checkpoint(epoch: int, weights, bias) -> None:
        val_pred = intent_predict_labels(x_intent_val, weights, bias)
        val_f1 = macro_f1(intent_data["val"].labels, val_pred, len(INTENT_LABEL_TO_ID))
        write_checkpoint(
            checkpoint_dir=teacher_ckpt_dir,
            step=epoch,
            model_payload={
                "featureDim": args.intent_dim,
                "labelMap": ID_TO_INTENT_LABEL,
                "weights": weights.tolist(),
                "bias": bias.tolist(),
            },
            metric_name="val_macro_f1",
            metric_value=val_f1,
        )

    teacher_intent_w, teacher_intent_b, teacher_epochs_ran = train_intent_softmax(
        x_intent_train,
        intent_data["train"].labels,
        epochs=args.intent_epochs,
        lr=args.intent_lr,
        checkpoint_cadence_steps=args.checkpoint_cadence_steps,
        checkpoint_callback=_teacher_checkpoint,
    )

    teacher_probs_train = intent_predict_probs(x_intent_train, teacher_intent_w, teacher_intent_b)

    x_student_train = vectorize_texts(intent_data["train"].texts, args.student_intent_dim)
    x_student_val = vectorize_texts(intent_data["val"].texts, args.student_intent_dim)
    x_student_test = vectorize_texts(intent_data["test"].texts, args.student_intent_dim)

    def _student_checkpoint(epoch: int, weights, bias) -> None:
        val_pred = intent_predict_labels(x_student_val, weights, bias)
        val_f1 = macro_f1(intent_data["val"].labels, val_pred, len(INTENT_LABEL_TO_ID))
        write_checkpoint(
            checkpoint_dir=student_ckpt_dir,
            step=epoch,
            model_payload={
                "featureDim": args.student_intent_dim,
                "labelMap": ID_TO_INTENT_LABEL,
                "weights": weights.tolist(),
                "bias": bias.tolist(),
            },
            metric_name="val_macro_f1",
            metric_value=val_f1,
        )

    def _student_epoch(epoch: int, weights, bias) -> bool:
        nonlocal best_student_weights
        nonlocal best_student_bias

        val_pred = intent_predict_labels(x_student_val, weights, bias)
        val_f1 = macro_f1(intent_data["val"].labels, val_pred, len(INTENT_LABEL_TO_ID))

        metric_for_stop = val_f1

        live_summary = {
            "status": "running",
            "epoch": epoch,
            "metricPath": args.early_stop_metric,
            "student": {
                "intent": {
                    "val_macro_f1": val_f1,
                    "val_accuracy": accuracy(intent_data["val"].labels, val_pred),
                }
            },
            "earlyStop": {
                "enabled": bool(args.early_stop_enabled),
                "minDelta": float(args.early_stop_min_delta),
                "patience": int(args.early_stop_patience),
                "minEpochs": int(args.early_stop_min_epochs),
                "bestMetric": early_stop_state["bestMetric"],
                "bestEpoch": early_stop_state["bestEpoch"],
                "stalledEpochs": early_stop_state["stalledEpochs"],
                "stopped": early_stop_state["stopped"],
                "stopReason": early_stop_state["stopReason"],
            },
        }
        write_json(student_eval_summary_path, live_summary)

        metric_from_summary = read_metric_from_summary(
            student_eval_summary_path, args.early_stop_metric
        )
        if metric_from_summary is None:
            metric_from_summary = metric_for_stop

        best_metric = early_stop_state["bestMetric"]
        improved_after_read = best_metric is None or metric_from_summary > (
            float(best_metric) + float(args.early_stop_min_delta)
        )
        if improved_after_read:
            early_stop_state["bestMetric"] = metric_from_summary
            early_stop_state["bestEpoch"] = epoch
            early_stop_state["stalledEpochs"] = 0
            best_student_weights = weights.copy()
            best_student_bias = bias.copy()
        else:
            early_stop_state["stalledEpochs"] = int(early_stop_state["stalledEpochs"]) + 1

        live_summary["earlyStop"]["bestMetric"] = early_stop_state["bestMetric"]
        live_summary["earlyStop"]["bestEpoch"] = early_stop_state["bestEpoch"]
        live_summary["earlyStop"]["stalledEpochs"] = early_stop_state["stalledEpochs"]
        write_json(student_eval_summary_path, live_summary)

        if not args.early_stop_enabled:
            return True
        if epoch < args.early_stop_min_epochs:
            return True
        if int(early_stop_state["stalledEpochs"]) < args.early_stop_patience:
            return True

        early_stop_state["stopped"] = True
        early_stop_state["stopEpoch"] = epoch
        early_stop_state["stopReason"] = (
            f"no {args.early_stop_metric} improvement > {args.early_stop_min_delta} "
            f"for {args.early_stop_patience} epochs"
        )
        return False

    student_intent_w, student_intent_b, student_epochs_ran = train_intent_student(
        x_student_train,
        intent_data["train"].labels,
        teacher_probs_train,
        epochs=args.student_intent_epochs,
        lr=args.intent_lr,
        alpha=args.distill_alpha,
        checkpoint_cadence_steps=args.checkpoint_cadence_steps,
        checkpoint_callback=_student_checkpoint,
        epoch_callback=_student_epoch,
    )

    if best_student_weights is not None and best_student_bias is not None:
        student_intent_w = best_student_weights
        student_intent_b = best_student_bias

    teacher_intent_val_pred = intent_predict_labels(
        x_intent_val, teacher_intent_w, teacher_intent_b
    )
    teacher_intent_test_pred = intent_predict_labels(
        x_intent_test, teacher_intent_w, teacher_intent_b
    )
    student_intent_val_pred = intent_predict_labels(
        x_student_val, student_intent_w, student_intent_b
    )
    student_intent_test_pred = intent_predict_labels(
        x_student_test, student_intent_w, student_intent_b
    )

    x_axis_train = vectorize_texts(axis_data["train"].texts, args.intent_dim)
    x_axis_val = vectorize_texts(axis_data["val"].texts, args.intent_dim)
    x_axis_test = vectorize_texts(axis_data["test"].texts, args.intent_dim)

    y_axis_train = np.asarray(axis_data["train"].targets, dtype=np.float32)
    y_axis_val = np.asarray(axis_data["val"].targets, dtype=np.float32)
    y_axis_test = np.asarray(axis_data["test"].targets, dtype=np.float32)

    teacher_axis_w, teacher_axis_b = train_axis_ridge(x_axis_train, y_axis_train, l2=args.axis_l2)
    teacher_axis_train_pred = axis_predict(x_axis_train, teacher_axis_w, teacher_axis_b)

    x_axis_student_train = vectorize_texts(axis_data["train"].texts, args.student_intent_dim)
    x_axis_student_val = vectorize_texts(axis_data["val"].texts, args.student_intent_dim)
    x_axis_student_test = vectorize_texts(axis_data["test"].texts, args.student_intent_dim)

    blended_axis_targets = (args.distill_alpha * y_axis_train) + (
        (1.0 - args.distill_alpha) * teacher_axis_train_pred
    )
    student_axis_w, student_axis_b = train_axis_ridge(
        x_axis_student_train, blended_axis_targets, l2=args.axis_l2
    )

    teacher_axis_val_pred = axis_predict(x_axis_val, teacher_axis_w, teacher_axis_b)
    teacher_axis_test_pred = axis_predict(x_axis_test, teacher_axis_w, teacher_axis_b)
    student_axis_val_pred = axis_predict(x_axis_student_val, student_axis_w, student_axis_b)
    student_axis_test_pred = axis_predict(x_axis_student_test, student_axis_w, student_axis_b)

    teacher_metrics = {
        "intent": {
            "val_accuracy": accuracy(intent_data["val"].labels, teacher_intent_val_pred),
            "test_accuracy": accuracy(intent_data["test"].labels, teacher_intent_test_pred),
            "val_macro_f1": macro_f1(
                intent_data["val"].labels, teacher_intent_val_pred, len(INTENT_LABEL_TO_ID)
            ),
            "test_macro_f1": macro_f1(
                intent_data["test"].labels, teacher_intent_test_pred, len(INTENT_LABEL_TO_ID)
            ),
        },
        "axis": {
            "val_mse": mean_squared_error(y_axis_val, teacher_axis_val_pred),
            "test_mse": mean_squared_error(y_axis_test, teacher_axis_test_pred),
        },
    }

    student_metrics = {
        "intent": {
            "val_accuracy": accuracy(intent_data["val"].labels, student_intent_val_pred),
            "test_accuracy": accuracy(intent_data["test"].labels, student_intent_test_pred),
            "val_macro_f1": macro_f1(
                intent_data["val"].labels, student_intent_val_pred, len(INTENT_LABEL_TO_ID)
            ),
            "test_macro_f1": macro_f1(
                intent_data["test"].labels, student_intent_test_pred, len(INTENT_LABEL_TO_ID)
            ),
        },
        "axis": {
            "val_mse": mean_squared_error(y_axis_val, student_axis_val_pred),
            "test_mse": mean_squared_error(y_axis_test, student_axis_test_pred),
        },
    }

    teacher_artifacts = {
        "createdAt": utc_now_iso(),
        "intentRouter": {
            "featureDim": args.intent_dim,
            "labelMap": ID_TO_INTENT_LABEL,
            "weights": teacher_intent_w.tolist(),
            "bias": teacher_intent_b.tolist(),
        },
        "axisProposer": {
            "featureDim": args.intent_dim,
            "axisNames": axis_names,
            "weights": teacher_axis_w.tolist(),
            "bias": teacher_axis_b.tolist(),
        },
        "metrics": teacher_metrics,
    }

    student_artifacts = {
        "createdAt": utc_now_iso(),
        "intentRouter": {
            "featureDim": args.student_intent_dim,
            "labelMap": ID_TO_INTENT_LABEL,
            "weights": student_intent_w.tolist(),
            "bias": student_intent_b.tolist(),
            "distilledFrom": "teacher.intentRouter",
        },
        "axisProposer": {
            "featureDim": args.student_intent_dim,
            "axisNames": axis_names,
            "weights": student_axis_w.tolist(),
            "bias": student_axis_b.tolist(),
            "distilledFrom": "teacher.axisProposer",
        },
        "metrics": student_metrics,
    }

    teacher_bundle_path = teacher_dir / "model_bundle.json"
    student_bundle_path = student_dir / "model_bundle.json"
    write_model_artifact(teacher_bundle_path, teacher_artifacts)
    write_model_artifact(student_bundle_path, student_artifacts)
    write_json(
        teacher_dir / "eval_summary.json", {"status": "complete", "teacher": teacher_metrics}
    )
    write_json(
        student_dir / "eval_summary.json",
        {
            "status": "complete",
            "student": student_metrics,
            "earlyStop": {
                "enabled": bool(args.early_stop_enabled),
                "metricPath": args.early_stop_metric,
                "patience": args.early_stop_patience,
                "minDelta": args.early_stop_min_delta,
                "minEpochs": args.early_stop_min_epochs,
                "bestMetric": early_stop_state.get("bestMetric"),
                "bestEpoch": early_stop_state.get("bestEpoch"),
                "stopped": bool(early_stop_state.get("stopped")),
                "stopEpoch": int(early_stop_state.get("stopEpoch", 0)),
                "stopReason": str(early_stop_state.get("stopReason", "")),
            },
        },
    )

    teacher_prune = prune_checkpoints(
        checkpoint_dir=teacher_ckpt_dir,
        keep_last_n=args.max_checkpoints,
        metric_name="val_macro_f1",
        higher_is_better=True,
        final_export_path=teacher_bundle_path,
        max_total_checkpoints=args.max_total_checkpoints,
    )
    student_prune = prune_checkpoints(
        checkpoint_dir=student_ckpt_dir,
        keep_last_n=args.max_checkpoints,
        metric_name="val_macro_f1",
        higher_is_better=True,
        final_export_path=student_bundle_path,
        max_total_checkpoints=args.max_total_checkpoints,
    )

    metrics_payload = {
        "runId": args.run_id,
        "createdAt": utc_now_iso(),
        "student": student_metrics,
        "distillation": {
            "alpha": args.distill_alpha,
            "intentStudentDim": args.student_intent_dim,
            "axisStudentDim": args.student_intent_dim,
        },
        "checkpointPolicy": {
            "cadenceSteps": args.checkpoint_cadence_steps,
            "maxCheckpoints": args.max_checkpoints,
            "maxTotalCheckpoints": args.max_total_checkpoints,
            "pruning": {"teacher": teacher_prune, "student": student_prune},
        },
        "trainingProgress": {
            "teacherIntentEpochsRan": teacher_epochs_ran,
            "studentIntentEpochsRan": student_epochs_ran,
        },
        "earlyStop": {
            "enabled": bool(args.early_stop_enabled),
            "metricPath": args.early_stop_metric,
            "patience": args.early_stop_patience,
            "minDelta": args.early_stop_min_delta,
            "minEpochs": args.early_stop_min_epochs,
            "bestMetric": early_stop_state.get("bestMetric"),
            "bestEpoch": early_stop_state.get("bestEpoch"),
            "stopped": bool(early_stop_state.get("stopped")),
            "stopEpoch": int(early_stop_state.get("stopEpoch", 0)),
            "stopReason": str(early_stop_state.get("stopReason", "")),
        },
    }
    write_json(upload_root / "metrics.json", metrics_payload)
    assert_teacher_prefix(upload_root)

    output_uri = args.output_s3_uri.rstrip("/") + f"/{args.run_id}"
    uploaded_files = upload_directory_to_s3(
        s3_client=s3_client, local_root=upload_root, s3_uri=output_uri
    )

    result = {
        "runId": args.run_id,
        "outputS3Uri": output_uri,
        "uploadedFiles": uploaded_files,
        "teacherMetrics": teacher_metrics,
        "studentMetrics": student_metrics,
        "earlyStop": metrics_payload["earlyStop"],
        "checkpointPolicy": metrics_payload["checkpointPolicy"],
    }

    write_json(workdir / "train_result.json", result)
    if args.auto_shutdown_on_complete and bool(early_stop_state.get("stopped")):
        subprocess.run(["shutdown", "-h", "now"], check=False)
    return result


def main() -> int:
    args = parse_args()

    if args.allow_non_aws:
        if os.environ.get("KELLY_ALLOW_NON_AWS_DEBUG") != "1":
            raise RuntimeError(
                "--allow-non-aws requires KELLY_ALLOW_NON_AWS_DEBUG=1 environment variable"
            )
        print("WARNING: AWS GPU enforcement bypassed (debug mode)", flush=True)

    if not args.run_id:
        args.run_id = f"run-{utc_now_iso().replace(':', '').replace('-', '')}".lower()

    contract = load_run_contract_optional(args.run_contract)
    if args.package_local_dir is None:
        args.package_s3_uri, args.output_s3_uri = resolve_training_uris(args, contract)
    else:
        if not args.output_s3_uri:
            run_bucket = str(run_contract_get(contract, "s3", "runBucket", default="")).strip()
            run_prefix = str(
                run_contract_get(contract, "s3", "runPrefix", default="training/runs")
            ).strip("/")
            if not run_bucket:
                raise ValueError("Missing run bucket: provide --output-s3-uri or set s3.runBucket")
            args.output_s3_uri = build_s3_uri(run_bucket, run_prefix)

    checkpoint_settings = resolve_checkpoint_settings(
        contract=contract,
        cadence_steps=args.checkpoint_cadence_steps,
        max_checkpoints=args.max_checkpoints,
    )
    args.checkpoint_cadence_steps = checkpoint_settings["cadenceSteps"]
    args.max_checkpoints = checkpoint_settings["maxCheckpoints"]
    args.max_total_checkpoints = resolve_max_total_checkpoints(args, contract)
    early_stop_settings = resolve_early_stop_settings(args, contract)
    args.early_stop_enabled = bool(early_stop_settings["enabled"])
    args.early_stop_patience = int(early_stop_settings["patience"])
    args.early_stop_min_delta = float(early_stop_settings["minDelta"])
    args.early_stop_min_epochs = int(early_stop_settings["minEpochs"])
    args.early_stop_metric = str(early_stop_settings["metricPath"])
    args.auto_shutdown_on_complete = bool(early_stop_settings["autoShutdown"])

    package_resolved: dict[str, Any]
    if args.package_local_dir is None:
        package_bucket, package_prefix = parse_s3_uri(args.package_s3_uri)
        package_resolved = {
            "bucket": package_bucket,
            "prefix": package_prefix,
            "uri": args.package_s3_uri,
        }
    else:
        package_resolved = {"localDir": str(args.package_local_dir.resolve())}
    output_bucket, output_prefix = parse_s3_uri(args.output_s3_uri)
    resolved_output_uri = args.output_s3_uri.rstrip("/") + f"/{args.run_id}"
    resolved_output_bucket, resolved_output_prefix = parse_s3_uri(resolved_output_uri)

    if args.dry_run:
        payload = {
            "mode": "dry-run",
            "runId": args.run_id,
            "resolved": {
                "package": package_resolved,
                "outputRoot": {
                    "bucket": output_bucket,
                    "prefix": output_prefix,
                    "uri": args.output_s3_uri,
                },
                "outputRun": {
                    "bucket": resolved_output_bucket,
                    "prefix": resolved_output_prefix,
                    "uri": resolved_output_uri,
                },
                "checkpointCadenceSteps": args.checkpoint_cadence_steps,
                "maxCheckpoints": args.max_checkpoints,
                "maxTotalCheckpoints": args.max_total_checkpoints,
                "checkpointPolicy": checkpoint_settings,
                "earlyStop": early_stop_settings,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not args.allow_non_aws:
        require_aws_gpu()

    result = run_training(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
