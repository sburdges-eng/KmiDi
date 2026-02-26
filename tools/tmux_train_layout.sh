#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=""
TRAIN_CMD=""
TRAIN_LOG=""
WORKDIR="/opt/listening-train/work"
CACHE_DIR=""
WATCHDOG_CMD=""
COST_CMD=""
ATTACH=0

usage() {
  cat <<'USAGE'
Usage: tmux_train_layout.sh --session-name <name> --train-cmd <command> [options]

Creates a tmux session with panes:
  TRAIN_LOG, GPU, IO/DISK, CACHE, CHECKPOINTS, WATCHDOG, COST

Options:
  --session-name <name>      tmux session name
  --train-cmd <command>      training command to run in TRAIN_LOG pane
  --train-log <path>         train log path
  --workdir <path>           workspace root (default: /opt/listening-train/work)
  --cache-dir <path>         cache directory to monitor
  --watchdog-cmd <command>   watchdog command
  --cost-cmd <command>       cost monitor command
  --attach                   attach immediately
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --train-cmd)
      TRAIN_CMD="$2"
      shift 2
      ;;
    --train-log)
      TRAIN_LOG="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --watchdog-cmd)
      WATCHDOG_CMD="$2"
      shift 2
      ;;
    --cost-cmd)
      COST_CMD="$2"
      shift 2
      ;;
    --attach)
      ATTACH=1
      shift
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

if [[ -z "${SESSION_NAME}" || -z "${TRAIN_CMD}" ]]; then
  echo "--session-name and --train-cmd are required" >&2
  usage >&2
  exit 2
fi

if [[ -z "${TRAIN_LOG}" ]]; then
  TRAIN_LOG="${WORKDIR}/logs/train.log"
fi
mkdir -p "${WORKDIR}/logs"
mkdir -p "$(dirname "${TRAIN_LOG}")"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required on the training host" >&2
  exit 2
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 2
fi

TRAIN_RUNNER="${WORKDIR}/run_train_command.sh"
cat > "${TRAIN_RUNNER}" <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
bash -lc "${TRAIN_CMD}" 2>&1 | tee -a "${TRAIN_LOG}"
status=\${PIPESTATUS[0]}
echo "\${status}" > "${WORKDIR}/train.exit"
touch "${WORKDIR}/train_complete.signal"
exit "\${status}"
RUNNER
chmod +x "${TRAIN_RUNNER}"

CHECKPOINT_VIEW_CMD="while true; do date -u; echo; ls -lh '${WORKDIR}/upload_bundle/artifacts/student/checkpoints' 2>/dev/null || true; echo; ls -lh '${WORKDIR}/upload_bundle/artifacts/teacher/checkpoints' 2>/dev/null || true; sleep 5; clear; done"
IO_VIEW_CMD="while true; do date -u; echo; df -h; echo; (iostat -dx 1 2 || vmstat 1 2 || true); sleep 5; clear; done"
GPU_VIEW_CMD="while true; do date -u; echo; nvidia-smi || true; sleep 2; clear; done"
CACHE_VIEW_CMD="while true; do date -u; echo; if [[ -n '${CACHE_DIR}' ]]; then du -sh '${CACHE_DIR}' 2>/dev/null || true; ls -lah '${CACHE_DIR}' 2>/dev/null | head -n 40 || true; else echo 'cache directory not set'; fi; sleep 5; clear; done"
WATCHDOG_VIEW_CMD="${WATCHDOG_CMD:-while true; do date -u; echo 'watchdog idle'; sleep 10; clear; done}"
COST_VIEW_CMD="${COST_CMD:-while true; do date -u; echo 'cost monitor idle'; sleep 15; clear; done}"

export TRAIN_CMD
export TRAIN_LOG
export WORKDIR

# Pane 0: TRAIN_LOG
_tmux_new_cmd="${TRAIN_RUNNER}"
tmux new-session -d -s "${SESSION_NAME}" -n train "bash -lc '${_tmux_new_cmd}'"

# Panes 1-6
tmux split-window -t "${SESSION_NAME}:0" -h "bash -lc \"${GPU_VIEW_CMD}\""
tmux split-window -t "${SESSION_NAME}:0" -v "bash -lc \"${IO_VIEW_CMD}\""
tmux split-window -t "${SESSION_NAME}:0" -v "bash -lc \"${CACHE_VIEW_CMD}\""
tmux split-window -t "${SESSION_NAME}:0" -v "bash -lc \"${CHECKPOINT_VIEW_CMD}\""
tmux split-window -t "${SESSION_NAME}:0" -v "bash -lc \"${WATCHDOG_VIEW_CMD}\""
tmux split-window -t "${SESSION_NAME}:0" -v "bash -lc \"${COST_VIEW_CMD}\""

tmux select-layout -t "${SESSION_NAME}:0" tiled

# Pane titles for quick identification
tmux select-pane -t "${SESSION_NAME}:0.0" -T "TRAIN_LOG"
tmux select-pane -t "${SESSION_NAME}:0.1" -T "GPU"
tmux select-pane -t "${SESSION_NAME}:0.2" -T "IO/DISK"
tmux select-pane -t "${SESSION_NAME}:0.3" -T "CACHE"
tmux select-pane -t "${SESSION_NAME}:0.4" -T "CHECKPOINTS"
tmux select-pane -t "${SESSION_NAME}:0.5" -T "WATCHDOG"
tmux select-pane -t "${SESSION_NAME}:0.6" -T "COST"

echo "Created tmux session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"

if [[ ${ATTACH} -eq 1 ]]; then
  exec tmux attach -t "${SESSION_NAME}"
fi
