#!/usr/bin/env bash
# Launch cloud training and attach to monitor.
# Run from your Mac. SSHs to cloud instance, runs setup+train or attaches to tmux.
#
# Usage:
#   CLOUD_SSH_HOST=ubuntu@1.2.3.4 ./scripts/cloud_launch_and_monitor.sh [setup|train|monitor]
#   Or: ./scripts/cloud_launch_and_monitor.sh ubuntu@1.2.3.4 [setup|train|monitor]
#
# Modes:
#   setup   — One-time: ./scripts/cloud_setup.sh (clone, docker build)
#   train   — Launch ./scripts/cloud_train.sh in tmux
#   monitor — Attach to tmux session (default if no mode)
#   both    — setup, then train, then attach
#
# Reference: docs/CLOUD_SETUP_REFERENCE.md

set -e

SSH_HOST="${CLOUD_SSH_HOST:-$1}"
MODE="${2:-monitor}"
if [[ -n "$1" && "$1" != setup && "$1" != train && "$1" != monitor && "$1" != both ]]; then
  SSH_HOST="$1"
  MODE="${2:-monitor}"
elif [[ -z "$SSH_HOST" ]]; then
  echo "Usage: CLOUD_SSH_HOST=user@ip ./scripts/cloud_launch_and_monitor.sh [setup|train|monitor|both]"
  echo "   Or: ./scripts/cloud_launch_and_monitor.sh user@ip [setup|train|monitor|both]"
  echo ""
  echo "AWS: CLOUD_SSH_KEY=~/.ssh/key.pem (required for EC2)"
  echo "  export CLOUD_SSH_HOST=ubuntu@<ec2-ip>"
  echo "  export CLOUD_SSH_KEY=~/.ssh/your-key.pem"
  echo "  ./scripts/cloud_launch_and_monitor.sh both"
  exit 1
fi

REPO_URL="${CLOUD_REPO_URL:-https://github.com/sburdges-eng/KmiDi}"
SSH_KEY="${CLOUD_SSH_KEY:-}"

run_remote() {
  local ssh_opts="-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
  [[ -n "$SSH_KEY" && -f "$SSH_KEY" ]] && ssh_opts="$ssh_opts -i $SSH_KEY"
  ssh $ssh_opts "$SSH_HOST" "$@"
}

case "$MODE" in
  setup)
    echo "[setup] Running cloud_setup.sh on $SSH_HOST..."
    run_remote "cd /workspace/repos/kmidi 2>/dev/null || (mkdir -p /workspace/repos && git clone $REPO_URL /workspace/repos/kmidi) && cd /workspace/repos/kmidi && ./scripts/cloud_setup.sh $REPO_URL"
    ;;
  train)
    echo "[train] Launching cloud_train.sh on $SSH_HOST..."
    run_remote "cd /workspace/repos/kmidi && ./scripts/cloud_train.sh"
    ;;
  monitor)
    echo "[monitor] Attaching to tmux session kmidi-train on $SSH_HOST (Ctrl+B D to detach)..."
    MONITOR_OPTS="-t -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
    [[ -n "$SSH_KEY" && -f "$SSH_KEY" ]] && MONITOR_OPTS="$MONITOR_OPTS -i $SSH_KEY"
    exec ssh $MONITOR_OPTS "$SSH_HOST" "tmux attach -t kmidi-train"
    ;;
  both)
    echo "[both] Setup + train + monitor..."
    run_remote "mkdir -p /workspace/repos && ( [ ! -d /workspace/repos/kmidi/.git ] && git clone $REPO_URL /workspace/repos/kmidi || true ) && cd /workspace/repos/kmidi && git pull && ./scripts/cloud_setup.sh $REPO_URL"
    run_remote "cd /workspace/repos/kmidi && ./scripts/cloud_train.sh"
    echo ""
    echo "Attaching to tmux..."
    MONITOR_OPTS="-t -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
    [[ -n "$SSH_KEY" && -f "$SSH_KEY" ]] && MONITOR_OPTS="$MONITOR_OPTS -i $SSH_KEY"
    exec ssh $MONITOR_OPTS "$SSH_HOST" "tmux attach -t kmidi-train"
    ;;
  *)
    echo "Unknown mode: $MODE (use setup, train, monitor, or both)"
    exit 1
    ;;
esac
