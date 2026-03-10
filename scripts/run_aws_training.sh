#!/usr/bin/env bash
# =============================================================================
# Fastest/optimized AWS training: run on GPU instance, minimal args, run contract
# =============================================================================
# Prereqs: config/run_contract.yaml has s3.packageBucket, s3.packagePrefix,
#          training.activePackageId (or pass --package-s3-uri).
# Fastest: pre-cache package then use --package-local-dir (no S3 download).
# Multi-node: RANK=0 WORLD_SIZE=2 (etc.) before this script.
#
# Usage:
#   ./scripts/run_aws_training.sh
#   ./scripts/run_aws_training.sh --package-local-dir /path/to/PACKAGES/pkg-id
#   RANK=0 WORLD_SIZE=2 ./scripts/run_aws_training.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Run in tmux for long jobs: tmux new -s train"
exec python3 scripts/aws_train_entrypoint.py "$@"
