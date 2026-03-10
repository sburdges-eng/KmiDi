#!/usr/bin/env python3
"""SageMaker container entrypoint: read hyperparameters and run aws_train_entrypoint.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path("/opt/ml/code")
HP_CONFIG = Path("/opt/ml/input/config/hyperparameters.json")
PACKAGE_DIR = Path("/opt/ml/input/data/package")
RUN_CONTRACT = CODE_DIR / "config" / "run_contract.yaml"
ENTRYPOINT_SCRIPT = CODE_DIR / "scripts" / "aws_train_entrypoint.py"


def main() -> int:
    if not ENTRYPOINT_SCRIPT.exists():
        print(f"Entrypoint not found: {ENTRYPOINT_SCRIPT}", file=sys.stderr)
        return 1
    if not PACKAGE_DIR.exists():
        print(f"Package channel not found: {PACKAGE_DIR}", file=sys.stderr)
        return 1

    hyperparams = {}
    if HP_CONFIG.exists():
        try:
            hyperparams = json.loads(HP_CONFIG.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not read hyperparameters: {e}", file=sys.stderr)

    run_id = hyperparams.get("run_id", "run-sagemaker")
    output_s3_uri = hyperparams.get("output_s3_uri", "")
    workdir = hyperparams.get("workdir", "/opt/ml/output")

    if not output_s3_uri:
        print("Missing hyperparameter: output_s3_uri", file=sys.stderr)
        return 1

    argv = [
        str(ENTRYPOINT_SCRIPT),
        "--package-local-dir",
        str(PACKAGE_DIR),
        "--output-s3-uri",
        output_s3_uri,
        "--run-id",
        run_id,
        "--workdir",
        workdir,
    ]
    if RUN_CONTRACT.exists():
        argv.extend(["--run-contract", str(RUN_CONTRACT)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_DIR)

    result = subprocess.run(
        [sys.executable] + argv,
        env=env,
        cwd=str(CODE_DIR),
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
