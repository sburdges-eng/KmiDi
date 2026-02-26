#!/usr/bin/env python3
"""Integration tests for --dry-run paths across all training pipeline scripts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run_script(module: str, extra_args: list[str], env_override: dict[str, str] | None = None) -> dict:
    """Run a script module with --dry-run and return parsed JSON output."""
    cmd = [PYTHON, "-m", module, "--dry-run"] + extra_args
    env = None
    if env_override:
        import os
        env = {**os.environ, **env_override}
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, (
        f"{module} --dry-run failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    output = result.stdout.strip()
    assert output, f"{module} --dry-run produced no output"
    payload = json.loads(output)
    assert isinstance(payload, dict), f"{module} --dry-run did not return JSON object"
    return payload


def _make_test_contract(tmp: Path, bucket: str = "test-bucket", region: str = "us-east-1") -> Path:
    """Create a temporary run contract with populated values for testing."""
    contract = {
        "schemaVersion": "1.1",
        "s3": {
            "packageBucket": bucket,
            "packagePrefix": "training/packages",
            "runBucket": bucket,
            "runPrefix": "training/runs",
            "teacherPrefix": "artifacts/teacher",
            "studentPrefix": "artifacts/student",
        },
        "aws": {
            "region": region,
            "profile": "",
            "breakGlassProfile": "teacher-breakglass",
        },
        "training": {
            "activePackageId": "test-pkg-001",
            "defaultRunIdPrefix": "run",
            "defaultInstanceType": "g5.xlarge",
            "defaultMaxRuntimeHours": 4.0,
            "budgetCapUsd": 100.0,
            "checkpoints": {
                "defaultCadenceSteps": 10,
                "minCadenceSteps": 1,
                "maxCadenceSteps": 1000,
                "defaultMaxCheckpoints": 3,
                "maxCheckpoints": 20,
                "maxTotalCheckpoints": 21,
            },
            "earlyStop": {
                "enabled": True,
                "metricPath": "student.intent.val_macro_f1",
                "patience": 6,
                "minDelta": 0.001,
                "minEpochs": 8,
                "autoShutdownDefault": False,
            },
            "ops": {
                "preflightCommand": "./scripts/security_compliance_preflight.sh",
                "tmuxSessionPrefix": "train",
                "datasetCache": {
                    "defaultMode": "auto",
                    "ebsFallbackRoot": "/var/tmp/listening-cache",
                    "allowStreamingFallback": True,
                },
            },
        },
        "security": {
            "teacherFetch": {
                "allowByDefault": False,
                "requiredProfile": "teacher-breakglass",
                "requiredRoleArn": "arn:aws:iam::123456789012:role/teacher-breakglass",
            },
        },
    }
    contract_path = tmp / "run_contract.yaml"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract_path


class TestPackageDatasetDryRun:
    def test_dry_run_returns_valid_json(self, tmp_path: Path) -> None:
        contract_path = _make_test_contract(tmp_path)
        payload = _run_script(
            "scripts.package_dataset",
            [
                "--package-id", "test-pkg-dry",
                "--run-contract", str(contract_path),
            ],
        )
        assert payload["mode"] == "dry-run"
        assert "packageId" in payload
        assert payload["packageId"] == "test-pkg-dry"


class TestSyncPackageDryRun:
    def test_dry_run_returns_valid_json(self, tmp_path: Path) -> None:
        contract_path = _make_test_contract(tmp_path)
        # Create a minimal package dir with a task manifest
        pkg_dir = tmp_path / "pkg-dry"
        task_dir = pkg_dir / "intent_router"
        task_dir.mkdir(parents=True)
        manifest = {
            "schemaVersion": "1.0",
            "task": "intent_router",
            "createdAt": "2026-01-01T00:00:00Z",
            "hashAlgorithm": "sha256",
            "compression": "identity",
            "splits": {},
            "totals": {"recordCount": 0, "shardCount": 0},
        }
        (task_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        payload = _run_script(
            "scripts.sync_package_to_s3",
            [
                "--package-dir", str(pkg_dir),
                "--run-contract", str(contract_path),
            ],
        )
        assert payload["mode"] == "dry-run"
        assert payload["packageId"] == "pkg-dry"
        assert "resolved" in payload
        assert payload["resolved"]["bucket"] == "test-bucket"


class TestLaunchAwsTrainDryRun:
    def test_dry_run_returns_valid_json(self, tmp_path: Path) -> None:
        contract_path = _make_test_contract(tmp_path)
        payload = _run_script(
            "scripts.launch_aws_train",
            [
                "--package-id", "test-pkg-001",
                "--ami-id", "ami-placeholder",
                "--subnet-id", "subnet-placeholder",
                "--security-group-id", "sg-placeholder",
                "--iam-instance-profile", "placeholder-role",
                "--run-contract", str(contract_path),
                "--skip-preflight",
            ],
        )
        assert payload["mode"] == "dry-run"
        assert "runId" in payload
        assert payload["resolved"]["package"]["bucket"] == "test-bucket"
        assert payload["budgetCapUsd"] == 100.0


class TestAwsTrainEntrypointDryRun:
    def test_dry_run_returns_valid_json(self, tmp_path: Path) -> None:
        contract_path = _make_test_contract(tmp_path)
        payload = _run_script(
            "scripts.aws_train_entrypoint",
            [
                "--package-s3-uri", "s3://test-bucket/training/packages/test-pkg-001",
                "--output-s3-uri", "s3://test-bucket/training/runs",
                "--run-id", "test-dry-run",
                "--run-contract", str(contract_path),
            ],
        )
        assert payload["mode"] == "dry-run"
        assert payload["runId"] == "test-dry-run"
        assert "resolved" in payload


class TestFetchAwsArtifactsDryRun:
    def test_student_dry_run_returns_valid_json(self, tmp_path: Path) -> None:
        contract_path = _make_test_contract(tmp_path)
        payload = _run_script(
            "scripts.fetch_aws_artifacts",
            [
                "--s3-uri", "s3://test-bucket/training/runs/test-run",
                "--scope", "student",
                "--run-contract", str(contract_path),
            ],
        )
        assert payload["mode"] == "dry-run"
        assert payload["resolved"]["scope"] == "student"
        assert payload["resolved"]["bucket"] == "test-bucket"


class TestAllowNonAwsGate:
    def test_allow_non_aws_without_env_var_fails(self, tmp_path: Path) -> None:
        contract_path = _make_test_contract(tmp_path)
        cmd = [
            PYTHON, "-m", "scripts.aws_train_entrypoint",
            "--allow-non-aws",
            "--dry-run",
            "--package-s3-uri", "s3://test-bucket/training/packages/test-pkg",
            "--output-s3-uri", "s3://test-bucket/training/runs",
            "--run-id", "test-gate",
            "--run-contract", str(contract_path),
        ]
        import os
        env = {k: v for k, v in os.environ.items() if k != "KELLY_ALLOW_NON_AWS_DEBUG"}
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=30, env=env)
        assert result.returncode != 0
        assert "KELLY_ALLOW_NON_AWS_DEBUG" in result.stderr or "KELLY_ALLOW_NON_AWS_DEBUG" in result.stdout


class TestTeacherFetchRoleArnRequired:
    def test_empty_role_arn_rejects_teacher_fetch(self, tmp_path: Path) -> None:
        # Create contract with empty requiredRoleArn
        contract = {
            "schemaVersion": "1.1",
            "s3": {
                "packageBucket": "test-bucket",
                "packagePrefix": "training/packages",
                "runBucket": "test-bucket",
                "runPrefix": "training/runs",
            },
            "aws": {"region": "us-east-1", "profile": "default", "breakGlassProfile": "teacher-breakglass"},
            "training": {
                "defaultRunIdPrefix": "unit",
                "defaultInstanceType": "ml.g5.2xlarge",
                "defaultMaxRuntimeHours": 1.0,
                "budgetCapUsd": 10.0,
                "checkpoints": {
                    "defaultCadenceSteps": 100,
                    "minCadenceSteps": 10,
                    "maxCadenceSteps": 1000,
                    "defaultMaxCheckpoints": 3,
                    "maxCheckpoints": 10,
                    "maxTotalCheckpoints": 16,
                },
                "earlyStop": {
                    "enabled": True,
                    "metricPath": "metrics/val_loss",
                    "patience": 2,
                    "minDelta": 0.0,
                    "minEpochs": 0,
                    "autoShutdownDefault": False,
                },
                "ops": {
                    "preflightCommand": "echo preflight",
                    "tmuxSessionPrefix": "kmidi",
                    "datasetCache": {
                        "defaultMode": "auto",
                        "ebsFallbackRoot": "/tmp",
                        "allowStreamingFallback": False,
                    },
                },
            },
            "security": {
                "teacherFetch": {
                    "allowByDefault": False,
                    "requiredProfile": "teacher-breakglass",
                    "requiredRoleArn": "",
                }
            },
        }
        contract_path = tmp_path / "run_contract.yaml"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")

        cmd = [
            PYTHON, "-m", "scripts.fetch_aws_artifacts",
            "--dry-run",
            "--scope", "teacher",
            "--break-glass",
            "--profile", "teacher-breakglass",
            "--break-glass-role-arn", "arn:aws:iam::123456789012:role/some-role",
            "--s3-uri", "s3://test-bucket/training/runs/test-run",
            "--run-contract", str(contract_path),
        ]
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "requiredRoleArn" in result.stderr or "requiredRoleArn" in result.stdout
