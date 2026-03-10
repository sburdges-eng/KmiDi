#!/usr/bin/env bash
# =============================================================================
# One-time command: launch AWS cloud training (EC2 GPU + S3).
# =============================================================================
# 1. Create env from example and fill required values:
#      cp config/env.ec2-s3.example .env.ec2-s3
#      # edit .env.ec2-s3: AMI, subnet, security group, IAM profile, S3 buckets, package ID
# 2. Run this script (from repo root or scripts/):
#      ./scripts/launch_aws_cloud_training.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="$REPO_ROOT/.env.ec2-s3"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  echo "  cp config/env.ec2-s3.example .env.ec2-s3"
  echo "  # edit .env.ec2-s3 with AMI, subnet, security group, IAM profile, S3 buckets, package ID"
  exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

# Required EC2
: "${KMIDI_AWS_AMI_ID:?Set KMIDI_AWS_AMI_ID in .env.ec2-s3}"
: "${KMIDI_AWS_SUBNET_ID:?Set KMIDI_AWS_SUBNET_ID in .env.ec2-s3}"
: "${KMIDI_AWS_SECURITY_GROUP_ID:?Set KMIDI_AWS_SECURITY_GROUP_ID in .env.ec2-s3}"
: "${KMIDI_AWS_IAM_INSTANCE_PROFILE:?Set KMIDI_AWS_IAM_INSTANCE_PROFILE in .env.ec2-s3}"

# Package/output: full URIs or bucket + package ID
if [[ -n "${KMIDI_PACKAGE_S3_URI:-}" && -n "${KMIDI_OUTPUT_S3_URI:-}" ]]; then
  PACKAGE_URI="$KMIDI_PACKAGE_S3_URI"
  OUTPUT_URI="$KMIDI_OUTPUT_S3_URI"
elif [[ -n "${KMIDI_PACKAGE_BUCKET:-}" && -n "${KMIDI_RUN_BUCKET:-}" && -n "${KMIDI_PACKAGE_ID:-}" ]]; then
  PACKAGE_URI="s3://${KMIDI_PACKAGE_BUCKET}/training/packages/${KMIDI_PACKAGE_ID}"
  OUTPUT_URI="s3://${KMIDI_RUN_BUCKET}/training/runs"
else
  echo "Set either (KMIDI_PACKAGE_S3_URI + KMIDI_OUTPUT_S3_URI) or (KMIDI_PACKAGE_BUCKET + KMIDI_RUN_BUCKET + KMIDI_PACKAGE_ID) in .env.ec2-s3"
  exit 1
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --ami-id "$KMIDI_AWS_AMI_ID"
  --subnet-id "$KMIDI_AWS_SUBNET_ID"
  --security-group-id "$KMIDI_AWS_SECURITY_GROUP_ID"
  --iam-instance-profile "$KMIDI_AWS_IAM_INSTANCE_PROFILE"
  --package-s3-uri "$PACKAGE_URI"
  --output-s3-uri "$OUTPUT_URI"
)

[[ -n "${KMIDI_AWS_KEY_NAME:-}" ]]          && ARGS+=(--key-name "$KMIDI_AWS_KEY_NAME")
if [[ -n "${KMIDI_AWS_REGION:-}" ]]; then
  ARGS+=(--region "$KMIDI_AWS_REGION")
elif [[ -n "${AWS_DEFAULT_REGION:-}" ]]; then
  ARGS+=(--region "$AWS_DEFAULT_REGION")
fi
if [[ -n "${KMIDI_AWS_PROFILE:-}" ]]; then
  ARGS+=(--profile "$KMIDI_AWS_PROFILE")
elif [[ -n "${AWS_PROFILE:-}" ]]; then
  ARGS+=(--profile "$AWS_PROFILE")
fi
[[ -n "${KMIDI_AWS_INSTANCE_TYPE:-}" ]]     && ARGS+=(--instance-type "$KMIDI_AWS_INSTANCE_TYPE")
[[ -n "${KMIDI_RUN_ID:-}" ]]                && ARGS+=(--run-id "$KMIDI_RUN_ID")
[[ "${KMIDI_DRY_RUN:-0}" = "1" ]]           && ARGS+=(--dry-run)

exec python3 scripts/launch_aws_train.py "${ARGS[@]}"
