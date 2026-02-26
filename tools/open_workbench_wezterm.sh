#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INSTANCE_ID=""
RUN_ID=""
SESSION_NAME="train"
PROFILE=""
REGION=""
SSH_HOST=""
SSH_USER="ec2-user"
SSH_KEY_PATH=""

usage() {
  cat <<'USAGE'
Usage: open_workbench_wezterm.sh --instance-id <id> [options]

Opens local terminal windows:
  TRAIN (remote attach), OPUS, CODEX, SPARK

Options:
  --instance-id <id>      EC2 instance id (required)
  --run-id <id>           Run id (for labels)
  --session-name <name>   tmux session name on remote host (default: train)
  --profile <profile>     AWS profile
  --region <region>       AWS region
  --ssh-host <host>       SSH fallback host for TRAIN pane
  --ssh-user <user>       SSH fallback user (default: ec2-user)
  --ssh-key-path <path>   SSH private key path
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-id)
      INSTANCE_ID="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
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

TRAIN_CMD="${ROOT}/tools/open_train_terminal.sh --instance-id ${INSTANCE_ID} --session-name ${SESSION_NAME}"
if [[ -n "${PROFILE}" ]]; then
  TRAIN_CMD+=" --profile ${PROFILE}"
fi
if [[ -n "${REGION}" ]]; then
  TRAIN_CMD+=" --region ${REGION}"
fi
if [[ -n "${SSH_HOST}" ]]; then
  TRAIN_CMD+=" --ssh-host ${SSH_HOST}"
fi
if [[ -n "${SSH_USER}" ]]; then
  TRAIN_CMD+=" --ssh-user ${SSH_USER}"
fi
if [[ -n "${SSH_KEY_PATH}" ]]; then
  TRAIN_CMD+=" --ssh-key-path ${SSH_KEY_PATH}"
fi

OPUS_CMD="claude --model claude-opus-4-6"
CODEX_CMD="codex --model gpt-5.3-codex"
SPARK_CMD="codex --model gpt-5.3-codex-spark"

if command -v wezterm >/dev/null 2>&1; then
  wezterm cli spawn --new-window --cwd "${ROOT}" -- zsh -lc "echo 'TRAIN ${RUN_ID}'; ${TRAIN_CMD}"
  wezterm cli spawn --new-window --cwd "${ROOT}" -- zsh -lc "echo 'OPUS ${RUN_ID}'; ${OPUS_CMD}"
  wezterm cli spawn --new-window --cwd "${ROOT}" -- zsh -lc "echo 'CODEX ${RUN_ID}'; ${CODEX_CMD}"
  wezterm cli spawn --new-window --cwd "${ROOT}" -- zsh -lc "echo 'SPARK ${RUN_ID}'; ${SPARK_CMD}"
  echo "Opened TRAIN/OPUS/CODEX/SPARK in WezTerm."
  exit 0
fi

echo "WezTerm not found. Run these Terminal.app fallback commands:"
echo "osascript -e 'tell application \"Terminal\" to do script \"cd ${ROOT}; ${TRAIN_CMD}\"'"
echo "osascript -e 'tell application \"Terminal\" to do script \"cd ${ROOT}; ${OPUS_CMD}\"'"
echo "osascript -e 'tell application \"Terminal\" to do script \"cd ${ROOT}; ${CODEX_CMD}\"'"
echo "osascript -e 'tell application \"Terminal\" to do script \"cd ${ROOT}; ${SPARK_CMD}\"'"
