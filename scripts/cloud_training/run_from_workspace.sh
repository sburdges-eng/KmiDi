#!/usr/bin/env bash
# =============================================================================
# KMidi cloud training: run from Cursor workspace, logs/status in repo for AI
# =============================================================================
# Primary target: Lambda Labs (A100). Use config/cloud_training_lambda.yaml for
# "as well trained as possible" (Melody 200+, Harmony 150+, MIDI 15-30 epochs).
#
# Usage:
#   ./scripts/cloud_training/run_from_workspace.sh ssh USER@HOST /remote/path [--config CONFIG]
#   ./scripts/cloud_training/run_from_workspace.sh lambda [--config CONFIG]   # uses LAMBDA_SSH
#   ./scripts/cloud_training/run_from_workspace.sh attach USER@HOST /remote/path
#   ./scripts/cloud_training/run_from_workspace.sh docker [--config CONFIG] [--gpus N]
#
# Lambda: set LAMBDA_SSH=ubuntu@<ip> then run with "lambda". Default config is
# config/cloud_training_lambda.yaml.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATUS_FILE="$REPO_ROOT/cloud_training/status.json"
LOG_FILE="$REPO_ROOT/cloud_training/logs/training_latest.log"
LAMBDA_DEFAULT_CONFIG="config/cloud_training_lambda.yaml"

write_status() {
  local state="$1"
  local message="${2:-}"
  local host="${3:-}"
  local path="${4:-}"
  mkdir -p "$REPO_ROOT/cloud_training/logs"
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf '{"state":"%s","message":"%s","remote_host":"%s","remote_path":"%s","started_at":"%s","updated_at":"%s","log_file":"cloud_training/logs/training_latest.log"}\n' \
    "$state" "$message" "$host" "$path" "${STARTED_AT:-$ts}" "$ts" > "$STATUS_FILE"
}

usage() {
  echo "Usage: $0 ssh USER@HOST /remote/path [--config CONFIG]"
  echo "       $0 lambda [--config CONFIG]     # uses LAMBDA_SSH, default config $LAMBDA_DEFAULT_CONFIG"
  echo "       $0 attach USER@HOST /remote/path"
  echo "       $0 docker [--config CONFIG] [--gpus N]"
  exit 1
}

STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
MODE="${1:-}"
shift || true

if [[ -z "$MODE" ]]; then
  usage
fi

cd "$REPO_ROOT"
mkdir -p cloud_training/logs

# Resolve Lambda: lambda -> ssh $LAMBDA_SSH /workspace/KmiDi [--config cloud_training_lambda.yaml]
if [[ "$MODE" == "lambda" ]]; then
  if [[ -z "${LAMBDA_SSH:-}" ]]; then
    echo "Error: LAMBDA_SSH not set. Example: LAMBDA_SSH=ubuntu@1.2.3.4 $0 lambda"
    exit 1
  fi
  CONFIG="$LAMBDA_DEFAULT_CONFIG"
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--config" && -n "${2:-}" ]]; then
      CONFIG="$2"
      shift 2
    else
      shift
    fi
  done
  exec "$0" ssh "$LAMBDA_SSH" "/workspace/KmiDi" --config "$CONFIG"
fi

if [[ "$MODE" == "attach" ]]; then
  REMOTE_HOST="${1:-}"
  REMOTE_PATH="${2:-/workspace/KmiDi}"
  if [[ -z "$REMOTE_HOST" ]]; then
    echo "Usage: $0 attach USER@HOST [/remote/path]"
    exit 1
  fi
  write_status "running" "attached (tail -f)" "$REMOTE_HOST" "$REMOTE_PATH"
  echo "Tailing $REMOTE_HOST:$REMOTE_PATH/training.log -> $LOG_FILE (Ctrl+C to stop tail only)"
  ssh -t "$REMOTE_HOST" "tail -f $REMOTE_PATH/training.log" 2>/dev/null | tee -a "$LOG_FILE" || true
  write_status "idle" "attach session ended" "$REMOTE_HOST" "$REMOTE_PATH"
  exit 0
fi

if [[ "$MODE" == "docker" ]]; then
  CONFIG="config/cloud_training.yaml"
  GPUS=1
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config) CONFIG="${2:-}"; shift 2 ;;
      --gpus)   GPUS="${2:-1}"; shift 2 ;;
      *) shift ;;
    esac
  done
  write_status "running" "docker (train_cloud.sh)" "" ""
  echo "Running Docker cloud training (config=$CONFIG, gpus=$GPUS)..."
  bash "$REPO_ROOT/scripts/train_cloud.sh" --config "$CONFIG" --gpus "$GPUS" 2>&1 | tee "$LOG_FILE" || true
  write_status "done" "docker run finished" "" ""
  exit 0
fi

if [[ "$MODE" != "ssh" ]]; then
  usage
fi

REMOTE_HOST="${1:-}"
REMOTE_PATH="${2:-/workspace/KmiDi}"
shift 2 || true
CONFIG=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--config" && -n "${2:-}" ]]; then
    CONFIG="$2"
    shift 2
  else
    EXTRA_ARGS+=("$1")
    shift
  fi
done

if [[ -z "$REMOTE_HOST" ]]; then
  echo "Usage: $0 ssh USER@HOST [/remote/path] [--config CONFIG]"
  exit 1
fi

write_status "syncing" "rsync to $REMOTE_HOST" "$REMOTE_HOST" "$REMOTE_PATH"
echo "--- KMidi Cloud Run (workspace) ---"
echo "Target: $REMOTE_HOST"
echo "Remote path: $REMOTE_PATH"
echo ""

echo "[1/4] Syncing project..."
rsync -avz --exclude '.git' --exclude 'node_modules' --exclude 'build' \
  --exclude 'datasets' --exclude 'checkpoints' --exclude 'exports' \
  "$REPO_ROOT/" "${REMOTE_HOST}:${REMOTE_PATH}/"

echo "[2/4] Preparing remote environment..."
ssh "$REMOTE_HOST" "cd ${REMOTE_PATH} && bash scripts/setup_ml_env.sh" || true

write_status "running" "training started (nohup + tail)" "$REMOTE_HOST" "$REMOTE_PATH"
echo "[3/4] Starting training on remote (logs -> $LOG_FILE)..."

# When --config is set, run train.py (for cloud_training_lambda.yaml / integrated); else train_cuda_v2.sh
if [[ -n "$CONFIG" ]]; then
  RUN_CMD="cd ${REMOTE_PATH} && nohup bash -c 'python scripts/train.py --config ${CONFIG} --distributed 2>&1' > training.log 2>&1 & sleep 2 && tail -f training.log"
else
  RUN_CMD="cd ${REMOTE_PATH} && nohup bash -c 'bash scripts/train_cuda_v2.sh ${EXTRA_ARGS[*]} 2>&1' > training.log 2>&1 & sleep 2 && tail -f training.log"
fi
ssh -t "$REMOTE_HOST" "$RUN_CMD" 2>/dev/null | tee "$LOG_FILE" || RC=$?

# If tail was interrupted (e.g. Ctrl+C), training keeps running on remote
if [[ "${RC:-0}" -ne 0 ]]; then
  write_status "running" "training continues on remote; use 'attach' to tail again" "$REMOTE_HOST" "$REMOTE_PATH"
  echo ""
  echo "Training continues on $REMOTE_HOST. To view logs again:"
  echo "  $0 attach $REMOTE_HOST $REMOTE_PATH"
  exit 0
fi

write_status "done" "training finished" "$REMOTE_HOST" "$REMOTE_PATH"
echo "[4/4] Done. Checkpoints on remote: ${REMOTE_PATH}/checkpoints/"
