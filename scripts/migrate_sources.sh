#!/bin/bash
# Safe Source File Migration Script
set -e

DRY_RUN=${1:-""}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_FILE="$PROJECT_ROOT/MIGRATION_BACKUP_LOG.md"
echo "# Migration Backup Log" > "$LOG_FILE"
echo "**Started:** $(date)" >> "$LOG_FILE"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
warn() { echo "[WARNING] $1" | tee -a "$LOG_FILE"; }
error() {
  echo "[ERROR] $1" | tee -a "$LOG_FILE"
  exit 1
}

checksum() {
  if command -v shasum &> /dev/null; then
    shasum -a 256 "$1" | cut -d' ' -f1
  elif command -v sha256sum &> /dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    echo "unknown"
  fi
}

safe_move() {
  local src="$1"
  local dst="$2"
  local desc="$3"

  if [ ! -f "$src" ]; then
    error "Source file not found: $src"
  fi

  if [ -f "$dst" ] && [ "$src" != "$dst" ]; then
    error "Destination file already exists: $dst"
  fi

  local src_checksum=$(checksum "$src")
  log "Moving $desc: $src -> $dst (checksum: $src_checksum)"

  if [ "$DRY_RUN" != "dry-run" ]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    local dst_checksum=$(checksum "$dst")

    if [ "$src_checksum" != "$dst_checksum" ]; then
      error "Checksum mismatch after copy!"
    fi
    log "  Verified: Checksums match"
  else
    log "  [DRY RUN] Would copy file"
  fi
}

log "Starting migration${DRY_RUN:+ (DRY RUN)}"

safe_move "KmiDi_FINAL/engine/src/plugin/PluginProcessor.cpp" "src/plugin/PluginProcessor.cpp" "PluginProcessor.cpp"
safe_move "KmiDi_FINAL/engine/src/plugin/PluginProcessor.h" "src/plugin/PluginProcessor.h" "PluginProcessor.h"
safe_move "KmiDi_FINAL/engine/src/plugin/PluginEditor.cpp" "src/plugin/PluginEditor.cpp" "PluginEditor.cpp"
safe_move "KmiDi_FINAL/engine/src/plugin/PluginEditor.h" "src/plugin/PluginEditor.h" "PluginEditor.h"
safe_move "KmiDi_FINAL/engine/src/plugin/PluginState.cpp" "src/plugin/PluginState.cpp" "PluginState.cpp"
safe_move "KmiDi_FINAL/engine/src/plugin/PluginState.h" "src/plugin/PluginState.h" "PluginState.h"
safe_move "KmiDi_FINAL/engine/src/gui/main.cpp" "src/gui/main.cpp" "main.cpp"
safe_move "KmiDi_FINAL/engine/src/gui/main_window.cpp" "src/gui/main_window.cpp" "main_window.cpp"
[ -f "KmiDi_FINAL/engine/src/gui/main_window.h" ] && safe_move "KmiDi_FINAL/engine/src/gui/main_window.h" "src/gui/main_window.h" "main_window.h"
safe_move "KmiDi_FINAL/engine/src/bridge/kelly_ffi.cpp" "src/bridge/kelly_ffi.cpp" "kelly_ffi.cpp"
safe_move "KmiDi_FINAL/engine/src/bridge/kelly_ffi.h" "src/bridge/kelly_ffi.h" "kelly_ffi.h"

log "Migration${DRY_RUN:+ (DRY RUN)} complete!"
echo "**Completed:** $(date)" >> "$LOG_FILE"
