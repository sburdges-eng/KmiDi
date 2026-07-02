#!/usr/bin/env bash
# =============================================================================
# Cloud dev environment setup for KmiDi (Lambda, SageMaker, EC2, SSH training).
# Run from repo root. Idempotent: does not overwrite existing env/.env.training.
# =============================================================================
# Usage:
#   ./scripts/dev-setup-cloud.sh         # cloud-only (env + dirs + .env.training)
#   ./scripts/dev-setup-cloud.sh --full # run base dev-setup.sh first, then cloud
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

RUN_FULL=false
for arg in "$@"; do
  if [[ "$arg" == "--full" ]]; then
    RUN_FULL=true
    break
  fi
done

if [[ "$RUN_FULL" == "true" ]]; then
  echo "=> Running base dev setup first..."
  "$SCRIPT_DIR/dev-setup.sh"
  echo ""
fi

echo "=> Cloud dev setup (env, cloud_training dirs, .env.training)..."

# 1) Ensure env/ exists
mkdir -p "$ROOT_DIR/env"
echo "   env/ OK"

# 2) Ensure cloud_training/logs exists (status.json and logs path used by run_from_workspace.sh)
mkdir -p "$ROOT_DIR/cloud_training/logs"
echo "   cloud_training/logs OK"

# 3) Copy env/.env.training.example -> env/.env.training only if .env.training does not exist
ENV_TRAINING="$ROOT_DIR/env/.env.training"
ENV_EXAMPLE="$ROOT_DIR/env/.env.training.example"
if [[ ! -f "$ENV_TRAINING" ]] && [[ -f "$ENV_EXAMPLE" ]]; then
  cp "$ENV_EXAMPLE" "$ENV_TRAINING"
  echo "   Created env/.env.training from example (edit with your LAMBDA_SSH etc.)"
elif [[ -f "$ENV_TRAINING" ]]; then
  echo "   env/.env.training already exists (unchanged)"
else
  echo "   WARNING: env/.env.training.example not found; create env/.env.training manually if needed"
fi

echo ""
echo "=> Cloud dev setup complete."
echo ""
echo "Next steps:"
echo "  1. Set LAMBDA_SSH (e.g. export LAMBDA_SSH=ubuntu@<ip>) or edit env/.env.training"
echo "  2. Load training env:  source scripts/load-env.sh training"
echo "  3. Run cloud training: ./scripts/cloud_training/run_from_workspace.sh lambda"
echo "     Or SSH:             ./scripts/cloud_training/run_from_workspace.sh ssh USER@HOST /remote/path [--config config/cloud_training_lambda.yaml]"
echo "     Or attach:          ./scripts/cloud_training/run_from_workspace.sh attach USER@HOST /remote/path"
echo "  4. Status and logs:    cloud_training/status.json, cloud_training/logs/training_latest.log"
echo ""
echo "Local training first (16 GB Mac / free tier): see docs/LOCAL_TRAINING_SETUP.md"
echo "Cloud prerequisites and SageMaker/EC2: docs/CLOUD_DEV_SETUP.md"
