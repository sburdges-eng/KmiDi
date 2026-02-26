#!/usr/bin/env bash

set -euo pipefail
set -o allexport

SESSION="training"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Optional overrides for environment-specific values.
TRANSCRIPT_JSONL="${TRANSCRIPT_JSONL:-/Volumes/<external-ssd>/vault/transcripts.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)}"
PACKAGE_ID="${PACKAGE_ID:-pkg-${RUN_ID}}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
AMI_ID="${AMI_ID:-ami-xxxxxxxx}"
SUBNET_ID="${SUBNET_ID:-subnet-xxxxxxxx}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-xxxxxxxx}"
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-listening-train-role}"
FETCH_OUTPUT_DIR="${FETCH_OUTPUT_DIR:-artifacts/download}"

TMUX_CMD="${TMUX_CMD:-tmux}"

if ! command -v "$TMUX_CMD" >/dev/null 2>&1; then
  echo "tmux not found. Install tmux first." >&2
  exit 1
fi

build_cmd="python3 scripts/build_training_records.py --transcript-jsonl ${TRANSCRIPT_JSONL}"
split_cmd="python3 scripts/split_dataset.py"
package_cmd="python3 scripts/package_dataset.py --package-id ${PACKAGE_ID}"
validate_cmd="python3 scripts/validate_package.py --package-dir PACKAGES/${PACKAGE_ID}"
sync_cmd="python3 scripts/sync_package_to_s3.py --package-dir PACKAGES/${PACKAGE_ID}"
launch_cmd="python3 scripts/launch_aws_train.py --package-id ${PACKAGE_ID} --instance-type ${INSTANCE_TYPE} --ami-id ${AMI_ID} --subnet-id ${SUBNET_ID} --security-group-id ${SECURITY_GROUP_ID} --iam-instance-profile ${IAM_INSTANCE_PROFILE}"
fetch_cmd="python3 scripts/fetch_aws_artifacts.py --run-id <run-id> --output-dir ${FETCH_OUTPUT_DIR}"

# Start detached and replace any existing session.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" -n "prep"
tmux send-keys -t "$SESSION:prep" "echo 'Run from project: $PROJECT_DIR'" C-m
tmux send-keys -t "$SESSION:prep" "export TRANSCRIPT_JSONL=${TRANSCRIPT_JSONL}" C-m
tmux send-keys -t "$SESSION:prep" "export PACKAGE_ID=${PACKAGE_ID}" C-m
tmux send-keys -t "$SESSION:prep" "echo '$build_cmd'" C-m
tmux send-keys -t "$SESSION:prep" "$build_cmd" C-m

tmux new-window -t "$SESSION" -n "split"
tmux send-keys -t "$SESSION:split" "cd \"$PROJECT_DIR\"" C-m
tmux send-keys -t "$SESSION:split" "export PACKAGE_ID=${PACKAGE_ID}" C-m
tmux send-keys -t "$SESSION:split" "echo '$split_cmd'" C-m
tmux send-keys -t "$SESSION:split" "$split_cmd" C-m

tmux new-window -t "$SESSION" -n "package_sync"
tmux send-keys -t "$SESSION:package_sync" "cd \"$PROJECT_DIR\"" C-m
tmux send-keys -t "$SESSION:package_sync" "export PACKAGE_ID=${PACKAGE_ID}" C-m
tmux send-keys -t "$SESSION:package_sync" "echo '$package_cmd'" C-m
tmux send-keys -t "$SESSION:package_sync" "$package_cmd" C-m
tmux send-keys -t "$SESSION:package_sync" "echo '$validate_cmd'" C-m
tmux send-keys -t "$SESSION:package_sync" "$validate_cmd" C-m
tmux send-keys -t "$SESSION:package_sync" "echo '$sync_cmd'" C-m
tmux send-keys -t "$SESSION:package_sync" "$sync_cmd" C-m

tmux new-window -t "$SESSION" -n "launch"
tmux send-keys -t "$SESSION:launch" "cd \"$PROJECT_DIR\"" C-m
tmux send-keys -t "$SESSION:launch" "export PACKAGE_ID=${PACKAGE_ID} INSTANCE_TYPE=${INSTANCE_TYPE} AMI_ID=${AMI_ID} SUBNET_ID=${SUBNET_ID} SECURITY_GROUP_ID=${SECURITY_GROUP_ID} IAM_INSTANCE_PROFILE=${IAM_INSTANCE_PROFILE}" C-m
tmux send-keys -t "$SESSION:launch" "echo '$launch_cmd'" C-m
tmux send-keys -t "$SESSION:launch" "$launch_cmd" C-m

tmux new-window -t "$SESSION" -n "monitor"
tmux split-window -h -t "$SESSION:monitor" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:monitor.1" "cd \"$PROJECT_DIR\"" C-m
tmux send-keys -t "$SESSION:monitor.1" "echo '$fetch_cmd'" C-m
tmux send-keys -t "$SESSION:monitor.1" "echo 'Replace <run-id> after launch output appears in launch pane.'" C-m
tmux send-keys -t "$SESSION:monitor.1" "echo \"Then run: python3 scripts/fetch_aws_artifacts.py --run-id <run-id> --output-dir ${FETCH_OUTPUT_DIR}\"" C-m
tmux send-keys -t "$SESSION:monitor.2" "cd \"$PROJECT_DIR\"" C-m
tmux send-keys -t "$SESSION:monitor.2" "echo 'Optional: tail logs in terminal opened by remote launcher, or use tools/' C-m
tmux send-keys -t "$SESSION:monitor.2" "echo 'Common local check: ls -1 artifacts/download'" C-m

# Optional: run an empty window to keep local status/notes.
tmux new-window -t "$SESSION" -n "notes"
tmux send-keys -t "$SESSION:notes" "cd \"$PROJECT_DIR\"" C-m
tmux send-keys -t "$SESSION:notes" "printf 'Set your IDs in env before re-run: AMI_ID, SUBNET_ID, SECURITY_GROUP_ID\n'" C-m
tmux send-keys -t "$SESSION:notes" "printf 'You can override defaults with env vars: PACKAGE_ID, TRANSCRIPT_JSONL, RUN_ID, INSTANCE_TYPE, IAM_INSTANCE_PROFILE\n'" C-m

tmux select-window -t "$SESSION:prep"
tmux attach -t "$SESSION"
