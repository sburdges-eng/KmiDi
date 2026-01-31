#!/usr/bin/env bash
# Attach to tmux session "kmidi", or create it with cwd = project root.
# Usage: ./scripts/tmux_kmidi.sh   (from repo root or anywhere)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if tmux has-session -t kmidi 2>/dev/null; then
  exec tmux attach -t kmidi
else
  exec tmux new-session -s kmidi -c "$PROJECT_ROOT"
fi
