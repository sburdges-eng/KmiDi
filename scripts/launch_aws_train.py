#!/usr/bin/env python3
"""Launch AWS GPU training with strict budget and runner bootstrap."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from scripts.training_common import (
    DEFAULT_RUN_CONTRACT_PATH,
    build_s3_uri,
    load_run_contract,
    normalize_boolish,
    parse_s3_uri,
    resolve_checkpoint_settings,
    run_contract_get,
    utc_now_iso,
)

GPU_INSTANCE_PREFIXES: tuple[str, ...] = ("g3", "g4", "g5", "g6", "p2", "p3", "p4", "p5", "p6")
ON_DEMAND_GPU_PRICES_USD: dict[str, float] = {
    "g4dn.xlarge": 0.526,
    "g5.xlarge": 1.006,
    "g5.2xlarge": 1.212,
    "g5.4xlarge": 1.624,
    "g6.xlarge": 0.95,
    "p3.2xlarge": 3.06,
}


def _prefix_is_under(prefix: str, root: str) -> bool:
    normalized_prefix = prefix.strip("/")
    normalized_root = root.strip("/")
    return normalized_prefix == normalized_root or normalized_prefix.startswith(f"{normalized_root}/")


def is_gpu_instance_type(instance_type: str) -> bool:
    lowered = instance_type.strip().lower()
    return lowered.startswith(GPU_INSTANCE_PREFIXES)


def resolve_hourly_cost(instance_type: str, override_hourly: float) -> float:
    if override_hourly > 0:
        return override_hourly
    try:
        return ON_DEMAND_GPU_PRICES_USD[instance_type]
    except KeyError as exc:
        known = ", ".join(sorted(ON_DEMAND_GPU_PRICES_USD.keys()))
        raise ValueError(
            f"Unknown hourly price for {instance_type}. Provide --hourly-usd. Known: {known}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch AWS GPU training run with $100 hard cap.")
    parser.add_argument(
        "--package-s3-uri",
        default="",
        help="S3 URI to packaged dataset root. Defaults to run-contract package bucket/prefix + package id.",
    )
    parser.add_argument(
        "--output-s3-uri",
        default="",
        help="S3 URI where artifacts are written. Defaults to run-contract run bucket/prefix.",
    )
    parser.add_argument("--package-id", default="", help="Package ID used when --package-s3-uri is omitted")
    parser.add_argument("--instance-type", default="", help="GPU EC2 instance type")
    parser.add_argument("--ami-id", required=True, help="AMI ID for the training host")
    parser.add_argument("--subnet-id", required=True, help="Subnet ID")
    parser.add_argument("--security-group-id", required=True, help="Security Group ID")
    parser.add_argument("--iam-instance-profile", required=True, help="IAM instance profile name or ARN")
    parser.add_argument("--key-name", default="", help="Optional EC2 key pair")
    parser.add_argument("--region", default="", help="Optional AWS region")
    parser.add_argument("--profile", default="", help="Optional AWS profile")
    parser.add_argument("--run-id", default="", help="Run identifier")
    parser.add_argument("--tmux-session-name", default="", help="Remote tmux session name")
    parser.add_argument("--max-runtime-hours", type=float, default=0.0)
    parser.add_argument("--hourly-usd", type=float, default=0.0, help="Override on-demand hourly cost")
    parser.add_argument("--budget-cap-usd", type=float, default=0.0)
    parser.add_argument("--runner-s3-uri", default="", help="S3 URI for uploaded runner scripts")
    parser.add_argument("--teacher-hf-model", default="", help="Optional HF teacher model ID")
    parser.add_argument("--teacher-hf-revision", default="main")
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
        help="Absolute checkpoint cap (0 uses run-contract default).",
    )
    parser.add_argument("--early-stop-enabled", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=-1.0)
    parser.add_argument("--early-stop-min-epochs", type=int, default=0)
    parser.add_argument("--early-stop-metric", default="")
    parser.add_argument(
        "--auto-shutdown-on-complete",
        action="store_true",
        help="Shut down instance after early-stop completion",
    )
    parser.add_argument(
        "--dataset-cache-mode",
        default="",
        choices=["", "auto", "nvme", "ebs", "stream"],
        help="Dataset cache mode on remote host (default from run contract)",
    )
    parser.add_argument("--dataset-cache-root", default="", help="Dataset cache root override")
    parser.add_argument(
        "--allow-streaming-fallback",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Allow dataset streaming fallback when cache copy fails (-1 uses run contract)",
    )
    parser.add_argument("--ssh-host", default="", help="SSH host fallback for local attach")
    parser.add_argument("--ssh-user", default="ec2-user", help="SSH user fallback for local attach")
    parser.add_argument("--ssh-key-path", default="", help="SSH private key for local attach")
    parser.add_argument("--no-open-workbench", action="store_true", help="Do not open local TRAIN/OPUS/CODEX/SPARK terminals")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight gate (not recommended)")
    parser.add_argument("--preflight-command", default="", help="Override preflight command")
    parser.add_argument(
        "--run-contract",
        type=Path,
        default=DEFAULT_RUN_CONTRACT_PATH,
        help="Run contract config path (YAML/JSON).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_user_data(
    runner_s3_uri: str,
    package_s3_uri: str,
    output_s3_uri: str,
    run_id: str,
    session_name: str,
    max_runtime_hours: float,
    teacher_hf_model: str,
    teacher_hf_revision: str,
    checkpoint_cadence_steps: int,
    max_checkpoints: int,
    max_total_checkpoints: int,
    early_stop: dict[str, Any],
    dataset_cache_mode: str,
    dataset_cache_root: str,
    allow_streaming_fallback: bool,
    hourly_usd: float,
    budget_cap_usd: float,
) -> str:
    q = shlex.quote

    teacher_args = ""
    if teacher_hf_model:
        teacher_args = (
            f" --teacher-hf-model {q(teacher_hf_model)}"
            f" --teacher-hf-revision {q(teacher_hf_revision)}"
        )

    early_stop_args = ""
    if bool(early_stop.get("enabled", False)):
        early_stop_args += " --early-stop-enabled"
    early_stop_args += f" --early-stop-patience {int(early_stop.get('patience', 0))}"
    early_stop_args += f" --early-stop-min-delta {float(early_stop.get('minDelta', 0.0))}"
    early_stop_args += f" --early-stop-min-epochs {int(early_stop.get('minEpochs', 0))}"
    metric_path = str(early_stop.get("metricPath", "student.intent.val_macro_f1"))
    early_stop_args += f" --early-stop-metric {q(metric_path)}"
    if bool(early_stop.get("autoShutdown", False)):
        early_stop_args += " --auto-shutdown-on-complete"

    timeout_sec = int(max_runtime_hours * 3600)
    runner_uri = runner_s3_uri.rstrip("/")
    streaming_flag = 1 if allow_streaming_fallback else 0
    cache_root_arg = f" --cache-root {q(dataset_cache_root)}" if dataset_cache_root else ""

    q_runner_uri = q(runner_uri)
    q_package_s3_uri = q(package_s3_uri)
    q_output_s3_uri = q(output_s3_uri)
    q_run_id = q(run_id)
    q_session_name = q(session_name)
    q_metric_path = q(metric_path)

    return f"""#!/bin/bash
set -euo pipefail

mkdir -p /opt/listening-train/scripts
mkdir -p /opt/listening-train/config
mkdir -p /opt/listening-train/tools
mkdir -p /opt/listening-train/work
mkdir -p /opt/listening-train/work/logs

python3 -m pip install --upgrade pip
python3 -m pip install numpy boto3 huggingface_hub zstandard

aws s3 cp {q_runner_uri}/scripts/aws_train_entrypoint.py /opt/listening-train/scripts/aws_train_entrypoint.py
aws s3 cp {q_runner_uri}/scripts/training_common.py /opt/listening-train/scripts/training_common.py
aws s3 cp {q_runner_uri}/scripts/license_gate.py /opt/listening-train/scripts/license_gate.py
aws s3 cp {q_runner_uri}/config/run_contract.yaml /opt/listening-train/config/run_contract.yaml

aws s3 cp {q_runner_uri}/tools/tmux_train_layout.sh /opt/listening-train/tools/tmux_train_layout.sh
aws s3 cp {q_runner_uri}/tools/precache_dataset.sh /opt/listening-train/tools/precache_dataset.sh
aws s3 cp {q_runner_uri}/tools/watchdog.sh /opt/listening-train/tools/watchdog.sh
aws s3 cp {q_runner_uri}/tools/cost_estimate.sh /opt/listening-train/tools/cost_estimate.sh
chmod +x /opt/listening-train/tools/*.sh

CACHE_RESULT=/opt/listening-train/work/cache/precache_result.json
/opt/listening-train/tools/precache_dataset.sh \
  --package-s3-uri {q_package_s3_uri} \
  --workdir /opt/listening-train/work \
  --result-json "$CACHE_RESULT" \
  --mode {q(dataset_cache_mode)}{cache_root_arg} \
  --allow-streaming-fallback {streaming_flag}

PACKAGE_LOCAL_DIR="$(python3 - <<'PY'
import json
from pathlib import Path
path = Path('/opt/listening-train/work/cache/precache_result.json')
if not path.exists():
    print('')
else:
    payload = json.loads(path.read_text(encoding='utf-8'))
    print(payload.get('packageLocalDir', ''))
PY
)"

TRAIN_CMD="PYTHONPATH=/opt/listening-train timeout {timeout_sec} python3 /opt/listening-train/scripts/aws_train_entrypoint.py --run-contract /opt/listening-train/config/run_contract.yaml --workdir /opt/listening-train/work --output-s3-uri {q_output_s3_uri} --run-id {q_run_id} --checkpoint-cadence-steps {checkpoint_cadence_steps} --max-checkpoints {max_checkpoints} --max-total-checkpoints {max_total_checkpoints}{early_stop_args}{teacher_args}"
if [[ -n "$PACKAGE_LOCAL_DIR" ]]; then
  TRAIN_CMD="$TRAIN_CMD --package-local-dir $PACKAGE_LOCAL_DIR"
else
  TRAIN_CMD="$TRAIN_CMD --package-s3-uri {q_package_s3_uri}"
fi

COMPLETE_SIGNAL=/opt/listening-train/work/train_complete.signal
WATCHDOG_CMD="/opt/listening-train/tools/watchdog.sh --eval-summary /opt/listening-train/work/upload_bundle/artifacts/student/eval_summary.json --metric-path {q_metric_path} --poll-seconds 15 --complete-signal $COMPLETE_SIGNAL --patience {int(early_stop.get('patience', 0))} --min-delta {float(early_stop.get('minDelta', 0.0))} --status-file /opt/listening-train/work/logs/watchdog.status"
COST_CMD="/opt/listening-train/tools/cost_estimate.sh --hourly-usd {hourly_usd} --budget-cap-usd {budget_cap_usd} --start-epoch $(date +%s) --poll-seconds 30 --complete-signal $COMPLETE_SIGNAL"

/opt/listening-train/tools/tmux_train_layout.sh \
  --session-name {q_session_name} \
  --train-cmd "$TRAIN_CMD" \
  --train-log /opt/listening-train/work/logs/train.log \
  --workdir /opt/listening-train/work \
  --cache-dir "$PACKAGE_LOCAL_DIR" \
  --watchdog-cmd "$WATCHDOG_CMD" \
  --cost-cmd "$COST_CMD"

echo "[launch_aws_train] remote tmux started: {q_session_name}"
echo "[launch_aws_train] attach with: tmux attach -t {q_session_name}"
"""


def upload_runner_scripts(s3_client: object, runner_s3_uri: str, run_contract_path: Path) -> str:
    bucket, prefix = parse_s3_uri(runner_s3_uri)
    uploads = {
        "scripts/aws_train_entrypoint.py": Path("scripts/aws_train_entrypoint.py"),
        "scripts/training_common.py": Path("scripts/training_common.py"),
        "scripts/license_gate.py": Path("scripts/license_gate.py"),
        "config/run_contract.yaml": run_contract_path,
        "tools/tmux_train_layout.sh": Path("tools/tmux_train_layout.sh"),
        "tools/precache_dataset.sh": Path("tools/precache_dataset.sh"),
        "tools/watchdog.sh": Path("tools/watchdog.sh"),
        "tools/cost_estimate.sh": Path("tools/cost_estimate.sh"),
    }

    for remote_name, local_path in uploads.items():
        if not local_path.exists():
            raise FileNotFoundError(f"Missing required runner file: {local_path}")
        key = f"{prefix}/{remote_name}" if prefix else remote_name
        s3_client.upload_file(str(local_path), bucket, key)

    return f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"


def build_instance_profile_argument(profile: str) -> dict[str, str]:
    lowered = profile.lower()
    if lowered.startswith("arn:aws:iam::"):
        return {"Arn": profile}
    return {"Name": profile}


def validate_s3_prefixes(package_s3_uri: str, output_s3_uri: str, runner_s3_uri: str) -> None:
    _, package_prefix = parse_s3_uri(package_s3_uri)
    output_bucket, output_prefix = parse_s3_uri(output_s3_uri)
    if not _prefix_is_under(package_prefix, "training/packages"):
        raise ValueError(
            f"Invalid --package-s3-uri prefix: {package_s3_uri}. "
            "Expected s3://<bucket>/training/packages/..."
        )
    if not _prefix_is_under(output_prefix, "training/runs"):
        raise ValueError(
            f"Invalid --output-s3-uri prefix: {output_s3_uri}. "
            "Expected s3://<bucket>/training/runs/..."
        )
    if runner_s3_uri:
        runner_bucket, runner_prefix = parse_s3_uri(runner_s3_uri)
        if runner_bucket != output_bucket:
            raise ValueError(
                f"--runner-s3-uri bucket {runner_bucket} must match output bucket {output_bucket}"
            )
        if not _prefix_is_under(runner_prefix, output_prefix.strip("/")):
            raise ValueError(
                f"--runner-s3-uri prefix {runner_s3_uri} must stay under --output-s3-uri {output_s3_uri}"
            )


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
        package_id = args.package_id.strip() or str(
            run_contract_get(contract, "training", "activePackageId", default="")
        ).strip()
        package_bucket = str(run_contract_get(contract, "s3", "packageBucket", default="")).strip()
        package_prefix_root = str(
            run_contract_get(contract, "s3", "packagePrefix", default="training/packages")
        ).strip("/")
        if not package_id:
            raise ValueError("Missing package id: provide --package-s3-uri, --package-id, or training.activePackageId")
        if not package_bucket:
            raise ValueError("Missing package bucket: provide --package-s3-uri or set s3.packageBucket")
        package_prefix = f"{package_prefix_root}/{package_id}" if package_prefix_root else package_id
        package_uri = build_s3_uri(package_bucket, package_prefix)

    output_uri = args.output_s3_uri.strip()
    if not output_uri:
        run_bucket = str(run_contract_get(contract, "s3", "runBucket", default="")).strip()
        run_prefix = str(run_contract_get(contract, "s3", "runPrefix", default="training/runs")).strip("/")
        if not run_bucket:
            raise ValueError("Missing run bucket: provide --output-s3-uri or set s3.runBucket")
        output_uri = build_s3_uri(run_bucket, run_prefix)

    return package_uri, output_uri


def resolve_early_stop_settings(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    policy = run_contract_get(contract, "training", "earlyStop", default={})
    if not isinstance(policy, dict):
        policy = {}

    enabled = args.early_stop_enabled or normalize_boolish(policy.get("enabled", False))
    patience = args.early_stop_patience if args.early_stop_patience > 0 else int(policy.get("patience", 0) or 0)
    min_delta = (
        args.early_stop_min_delta
        if args.early_stop_min_delta >= 0
        else float(policy.get("minDelta", 0.0) or 0.0)
    )
    min_epochs = args.early_stop_min_epochs if args.early_stop_min_epochs > 0 else int(policy.get("minEpochs", 0) or 0)
    metric_path = args.early_stop_metric.strip() or str(policy.get("metricPath", "student.intent.val_macro_f1"))
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


def resolve_cache_settings(args: argparse.Namespace, contract: dict[str, Any]) -> tuple[str, str, bool]:
    policy = run_contract_get(contract, "training", "ops", "datasetCache", default={})
    if not isinstance(policy, dict):
        policy = {}

    mode = args.dataset_cache_mode or str(policy.get("defaultMode", "auto"))
    if mode not in {"auto", "nvme", "ebs", "stream"}:
        raise ValueError(f"Unsupported dataset cache mode: {mode}")

    root = args.dataset_cache_root or str(policy.get("ebsFallbackRoot", "")).strip()
    allow_streaming = (
        bool(args.allow_streaming_fallback)
        if args.allow_streaming_fallback in {0, 1}
        else normalize_boolish(policy.get("allowStreamingFallback", True))
    )
    return mode, root, allow_streaming


def resolve_max_total_checkpoints(args: argparse.Namespace, contract: dict[str, Any], max_checkpoints: int) -> int:
    checkpoint_policy = run_contract_get(contract, "training", "checkpoints", default={})
    if not isinstance(checkpoint_policy, dict):
        checkpoint_policy = {}

    resolved = args.max_total_checkpoints if args.max_total_checkpoints > 0 else int(
        checkpoint_policy.get("maxTotalCheckpoints", max_checkpoints + 1) or (max_checkpoints + 1)
    )
    if resolved <= 0:
        raise ValueError("max total checkpoints must be > 0")
    if resolved < max_checkpoints + 1:
        raise ValueError(
            f"max total checkpoints {resolved} is too small for keep-last-N + best policy ({max_checkpoints + 1})"
        )
    return resolved


def run_preflight_gate(preflight_command: str, dry_run: bool) -> None:
    if not preflight_command:
        raise ValueError("Missing preflight command; set training.ops.preflightCommand or pass --preflight-command")
    if dry_run:
        return

    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(["bash", "-lc", preflight_command], cwd=repo_root, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Preflight failed (exit {result.returncode}): {preflight_command}")


def open_local_workbench(
    run_id: str,
    instance_id: str,
    session_name: str,
    profile: str,
    region: str,
    ssh_host: str,
    ssh_user: str,
    ssh_key_path: str,
) -> str:
    script_path = Path(__file__).resolve().parent.parent / "tools" / "open_workbench_wezterm.sh"
    if not script_path.exists():
        return ""

    cmd: list[str] = [
        str(script_path),
        "--instance-id",
        instance_id,
        "--run-id",
        run_id,
        "--session-name",
        session_name,
    ]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])
    if ssh_host:
        cmd.extend(["--ssh-host", ssh_host])
    if ssh_user:
        cmd.extend(["--ssh-user", ssh_user])
    if ssh_key_path:
        cmd.extend(["--ssh-key-path", ssh_key_path])

    subprocess.run(cmd, check=False)
    return " ".join(cmd)


def main() -> int:
    args = parse_args()
    contract = load_run_contract_optional(args.run_contract)

    args.instance_type = args.instance_type or str(
        run_contract_get(contract, "training", "defaultInstanceType", default="g5.xlarge")
    )
    args.max_runtime_hours = args.max_runtime_hours if args.max_runtime_hours > 0 else float(
        run_contract_get(contract, "training", "defaultMaxRuntimeHours", default=4.0)
    )
    args.budget_cap_usd = args.budget_cap_usd if args.budget_cap_usd > 0 else float(
        run_contract_get(contract, "training", "budgetCapUsd", default=100.0)
    )
    args.profile = args.profile or str(run_contract_get(contract, "aws", "profile", default="")).strip()
    args.region = args.region or str(run_contract_get(contract, "aws", "region", default="")).strip()

    package_s3_uri, output_s3_uri = resolve_training_uris(args, contract)
    args.package_s3_uri = package_s3_uri
    args.output_s3_uri = output_s3_uri

    checkpoint_settings = resolve_checkpoint_settings(
        contract=contract,
        cadence_steps=args.checkpoint_cadence_steps,
        max_checkpoints=args.max_checkpoints,
    )
    args.checkpoint_cadence_steps = checkpoint_settings["cadenceSteps"]
    args.max_checkpoints = checkpoint_settings["maxCheckpoints"]
    args.max_total_checkpoints = resolve_max_total_checkpoints(args, contract, args.max_checkpoints)

    early_stop_settings = resolve_early_stop_settings(args, contract)
    cache_mode, cache_root, allow_streaming_fallback = resolve_cache_settings(args, contract)

    run_id_prefix = str(run_contract_get(contract, "training", "defaultRunIdPrefix", default="run")).strip() or "run"
    run_id = args.run_id or f"{run_id_prefix}-{utc_now_iso().replace(':', '').replace('-', '')}".lower()

    tmux_prefix = str(run_contract_get(contract, "training", "ops", "tmuxSessionPrefix", default="train")).strip()
    tmux_prefix = tmux_prefix or "train"
    session_name = args.tmux_session_name.strip() or f"{tmux_prefix}-{run_id}"

    preflight_command = args.preflight_command.strip() or str(
        run_contract_get(contract, "training", "ops", "preflightCommand", default="")
    ).strip()
    if not args.skip_preflight:
        run_preflight_gate(preflight_command=preflight_command, dry_run=args.dry_run)

    if not is_gpu_instance_type(args.instance_type):
        raise ValueError(
            f"Non-GPU instance type requested: {args.instance_type}. "
            "Training is restricted to AWS GPU instances."
        )

    if args.budget_cap_usd <= 0:
        raise ValueError("--budget-cap-usd must be > 0")
    if args.budget_cap_usd > 100.0:
        raise ValueError("Budget cap cannot exceed $100")
    if args.max_runtime_hours <= 0:
        raise ValueError("--max-runtime-hours must be > 0")

    hourly_cost = resolve_hourly_cost(args.instance_type, args.hourly_usd)
    estimated_cost = hourly_cost * args.max_runtime_hours
    if estimated_cost > args.budget_cap_usd:
        raise ValueError(
            f"Estimated runtime cost ${estimated_cost:.2f} exceeds cap ${args.budget_cap_usd:.2f}"
        )
    if estimated_cost > 100.0:
        raise ValueError(f"Estimated runtime cost ${estimated_cost:.2f} exceeds hard cap $100.00")

    runner_s3_uri = args.runner_s3_uri or (args.output_s3_uri.rstrip("/") + f"/{run_id}/runner")
    validate_s3_prefixes(
        package_s3_uri=args.package_s3_uri,
        output_s3_uri=args.output_s3_uri,
        runner_s3_uri=runner_s3_uri,
    )

    package_bucket, package_prefix = parse_s3_uri(args.package_s3_uri)
    output_bucket, output_prefix = parse_s3_uri(args.output_s3_uri)
    runner_bucket, runner_prefix = parse_s3_uri(runner_s3_uri)

    if args.dry_run:
        payload = {
            "mode": "dry-run",
            "runId": run_id,
            "sessionName": session_name,
            "instanceType": args.instance_type,
            "estimatedCostUsd": round(estimated_cost, 2),
            "budgetCapUsd": args.budget_cap_usd,
            "resolved": {
                "package": {"bucket": package_bucket, "prefix": package_prefix, "uri": args.package_s3_uri},
                "output": {"bucket": output_bucket, "prefix": output_prefix, "uri": args.output_s3_uri},
                "runner": {"bucket": runner_bucket, "prefix": runner_prefix, "uri": runner_s3_uri},
                "profile": args.profile,
                "region": args.region,
                "checkpointCadenceSteps": args.checkpoint_cadence_steps,
                "maxCheckpoints": args.max_checkpoints,
                "maxTotalCheckpoints": args.max_total_checkpoints,
                "checkpointPolicy": checkpoint_settings,
                "earlyStop": early_stop_settings,
                "datasetCache": {
                    "mode": cache_mode,
                    "root": cache_root,
                    "allowStreamingFallback": allow_streaming_fallback,
                },
                "preflight": {
                    "enabled": not args.skip_preflight,
                    "command": preflight_command,
                },
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required for AWS launch") from exc

    session_kwargs: dict[str, Any] = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs)
    ec2_client = session.client("ec2", region_name=args.region or None)
    s3_client = session.client("s3", region_name=args.region or None)

    uploaded_runner_uri = upload_runner_scripts(
        s3_client=s3_client,
        runner_s3_uri=runner_s3_uri,
        run_contract_path=args.run_contract,
    )

    user_data = build_user_data(
        runner_s3_uri=uploaded_runner_uri,
        package_s3_uri=args.package_s3_uri,
        output_s3_uri=args.output_s3_uri,
        run_id=run_id,
        session_name=session_name,
        max_runtime_hours=args.max_runtime_hours,
        teacher_hf_model=args.teacher_hf_model,
        teacher_hf_revision=args.teacher_hf_revision,
        checkpoint_cadence_steps=args.checkpoint_cadence_steps,
        max_checkpoints=args.max_checkpoints,
        max_total_checkpoints=args.max_total_checkpoints,
        early_stop=early_stop_settings,
        dataset_cache_mode=cache_mode,
        dataset_cache_root=cache_root,
        allow_streaming_fallback=allow_streaming_fallback,
        hourly_usd=hourly_cost,
        budget_cap_usd=args.budget_cap_usd,
    )

    launch_args: dict[str, Any] = {
        "ImageId": args.ami_id,
        "InstanceType": args.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "SubnetId": args.subnet_id,
        "SecurityGroupIds": [args.security_group_id],
        "IamInstanceProfile": build_instance_profile_argument(args.iam_instance_profile),
        "UserData": user_data,
        "InstanceInitiatedShutdownBehavior": "terminate",
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"listening-train-{run_id}"},
                    {"Key": "ListeningTrainRunId", "Value": run_id},
                    {"Key": "TmuxSession", "Value": session_name},
                    {"Key": "BudgetCapUSD", "Value": f"{args.budget_cap_usd:.2f}"},
                    {"Key": "EstimatedCostUSD", "Value": f"{estimated_cost:.2f}"},
                ],
            }
        ],
    }
    if args.key_name:
        launch_args["KeyName"] = args.key_name

    response = ec2_client.run_instances(**launch_args)
    instances = response.get("Instances", [])
    if not instances:
        raise RuntimeError("EC2 run_instances returned no instances")

    instance_id = instances[0].get("InstanceId", "")
    workbench_cmd = ""
    if not args.no_open_workbench:
        workbench_cmd = open_local_workbench(
            run_id=run_id,
            instance_id=instance_id,
            session_name=session_name,
            profile=args.profile,
            region=args.region,
            ssh_host=args.ssh_host,
            ssh_user=args.ssh_user,
            ssh_key_path=args.ssh_key_path,
        )

    result = {
        "runId": run_id,
        "instanceId": instance_id,
        "sessionName": session_name,
        "instanceType": args.instance_type,
        "estimatedCostUsd": round(estimated_cost, 2),
        "budgetCapUsd": args.budget_cap_usd,
        "checkpointCadenceSteps": args.checkpoint_cadence_steps,
        "maxCheckpoints": args.max_checkpoints,
        "maxTotalCheckpoints": args.max_total_checkpoints,
        "earlyStop": early_stop_settings,
        "datasetCache": {
            "mode": cache_mode,
            "root": cache_root,
            "allowStreamingFallback": allow_streaming_fallback,
        },
        "packageS3Uri": args.package_s3_uri,
        "outputS3Uri": args.output_s3_uri,
        "runnerS3Uri": uploaded_runner_uri,
        "preflight": {
            "enabled": not args.skip_preflight,
            "command": preflight_command,
        },
    }
    if workbench_cmd:
        result["workbenchCommand"] = workbench_cmd

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
