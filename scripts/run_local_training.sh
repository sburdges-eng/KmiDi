#!/usr/bin/env bash
# =============================================================================
# Run local training TUI in the current terminal (iTerm2, WezTerm, or any).
# For best UX run this script from iTerm2 or WezTerm so the Rich TUI displays
# progress, speed, and model metrics.
#
# Usage (from repo root):
#   ./scripts/run_local_training.sh
#   ./scripts/run_local_training.sh --config config/jepa_training_local_mac.yaml --model chord_jepa
#
# Open in a new iTerm window (macOS):
#   ./scripts/run_local_training.sh --new-window iterm
# Open in a new WezTerm tab:
#   ./scripts/run_local_training.sh --new-window wezterm
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TUI_SCRIPT="$SCRIPT_DIR/run_local_training_tui.py"

NEW_WINDOW=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --new-window)
      NEW_WINDOW="${2:-iterm}"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

run_tui() {
  cd "$REPO_ROOT"
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python "$TUI_SCRIPT" "${ARGS[@]}"
  else
    python3 "$TUI_SCRIPT" "${ARGS[@]}"
  fi
}

if [[ "$NEW_WINDOW" == "iterm" ]]; then
  if command -v osascript >/dev/null 2>&1; then
    RUN_CMD="cd $REPO_ROOT && python3 $TUI_SCRIPT ${ARGS[*]}"
    osascript -e "tell application \"iTerm\" to create window with default profile" \
      -e "tell current session of current window of application \"iTerm\" to write text \"$RUN_CMD\" with newline"
    exit 0
  fi
  echo "iTerm not available; run in current terminal:" >&2
  run_tui
  exit 0
fi

if [[ "$NEW_WINDOW" == "wezterm" ]]; then
  if command -v wezterm >/dev/null 2>&1; then
    cd "$REPO_ROOT"
    wezterm start -- python3 "$TUI_SCRIPT" "${ARGS[@]}"
    exit 0
  fi
  echo "WezTerm not found; run in current terminal:" >&2
  run_tui
  exit 0
fi

run_tui
