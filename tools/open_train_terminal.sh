#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INSTANCE_ID=""
SESSION_NAME="train"
PROFILE=""
REGION=""
SSH_HOST=""
SSH_USER="ec2-user"
SSH_KEY_PATH=""
FORCE_SSH=0

usage() {
  cat <<'USAGE'
Usage: open_train_terminal.sh --instance-id <id> [options]

Preferred path: AWS SSM start-session.
Fallback path: SSH + tmux attach.

Options:
  --instance-id <id>      EC2 instance id (required)
  --session-name <name>   tmux session name (default: train)
  --profile <profile>     AWS profile
  --region <region>       AWS region
  --force-ssh             Skip SSM and use SSH directly
  --ssh-host <host>       Public/private host for SSH fallback
  --ssh-user <user>       SSH user (default: ec2-user)
  --ssh-key-path <path>   SSH private key path
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-id)
      INSTANCE_ID="$2"
      shift 2
      ;;
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --force-ssh)
      FORCE_SSH=1
      shift
      ;;
    --ssh-host)
      SSH_HOST="$2"
      shift 2
      ;;
    --ssh-user)
      SSH_USER="$2"
      shift 2
      ;;
    --ssh-key-path)
      SSH_KEY_PATH="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "--instance-id is required" >&2
  usage >&2
  exit 2
fi

AWS_ARGS=()
if [[ -n "${PROFILE}" ]]; then
  AWS_ARGS+=(--profile "${PROFILE}")
fi
if [[ -n "${REGION}" ]]; then
  AWS_ARGS+=(--region "${REGION}")
fi

run_ssm_attach() {
  if ! command -v aws >/dev/null 2>&1; then
    return 1
  fi
  if ! command -v session-manager-plugin >/dev/null 2>&1; then
    return 1
  fi

  local remote_cmd
  remote_cmd="tmux attach -t ${SESSION_NAME} || tmux ls"

  echo "[open_train_terminal] attempting SSM attach: instance=${INSTANCE_ID} session=${SESSION_NAME}"
  if aws "${AWS_ARGS[@]}" ssm start-session \
    --target "${INSTANCE_ID}" \
    --document-name "AWS-StartInteractiveCommand" \
    --parameters "command=${remote_cmd}"; then
    return 0
  fi

  return 1
}

run_ssh_attach() {
  if [[ -z "${SSH_HOST}" ]]; then
    echo "[open_train_terminal] SSH fallback requires --ssh-host" >&2
    return 1
  fi

  local ssh_cmd=(ssh -tt)
  if [[ -n "${SSH_KEY_PATH}" ]]; then
    ssh_cmd+=(-i "${SSH_KEY_PATH}")
  fi
  ssh_cmd+=("${SSH_USER}@${SSH_HOST}" "tmux attach -t ${SESSION_NAME} || tmux ls")

  echo "[open_train_terminal] using SSH fallback: ${SSH_USER}@${SSH_HOST}"
  exec "${ssh_cmd[@]}"
}

if [[ ${FORCE_SSH} -eq 0 ]]; then
  if run_ssm_attach; then
    exit 0
  fi
  echo "[open_train_terminal] SSM unavailable or failed; trying SSH fallback." >&2
fi

run_ssh_attach
