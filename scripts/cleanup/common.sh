#!/usr/bin/env bash
# Shared utilities for safe cleanup scripts (Phase 1 & 2).
# Usage: source "$(dirname "$0")/common.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
DRY_RUN=false
QUARANTINE_BASE=""

# Default roots: KmiDi-1 and KmiDi
ROOTS=(
  "/Users/seanburdges/KmiDi-1"
  "/Users/seanburdges/KmiDi"
  "/Users/seanburdges/venv"
  "/Users/seanburdges/My Mac"
  "/Users/seanburdges/MP3"
  "/Users/seanburdges/ml-training-suite"
  "/Users/seanburdges/Emotion_Scale_Library"
  "/Users/seanburdges/Emotion_Instrument_Library"
  "/Users/seanburdges/audio"
  "/Users/seanburdges/_sorted"
  "/Users/seanburdges/RECOVERY_OPS"
  "/Users/seanburdges/ml"
  "/Users/seanburdges/Documents"
  "/Users/seanburdges/Desktop"
)

log() { echo "[cleanup] $*"; }
log_err() { echo "[cleanup] ERROR: $*" >&2; }

init_quarantine() {
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  QUARANTINE_BASE="/Users/seanburdges/KmiDi-1/_QUARANTINE_${stamp}"
  if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$QUARANTINE_BASE"
    log "Quarantine base: $QUARANTINE_BASE"
  else
    log "[DRY-RUN] Would use quarantine: $QUARANTINE_BASE"
  fi
}

# Exclude paths inside virtualenvs
exclude_venv() {
  grep -v -E '/\.?venv[^/]*/|/venv/|/\.venv/' || true
}

run_mv() {
  local src=$1 dest=$2
  if [[ "$DRY_RUN" == "true" ]]; then
    log "[DRY-RUN] mv $(printf '%q' "$src") -> $(printf '%q' "$dest")"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  mv -n -- "$src" "$dest" 2> /dev/null || mv -n "$src" "$dest"
}

run_rmdir() {
  local d=$1
  if [[ "$DRY_RUN" == "true" ]]; then
    log "[DRY-RUN] rmdir $(printf '%q' "$d")"
    return 0
  fi
  rmdir "$d" 2> /dev/null || true
}

# Parse --dry-run from args (callers pass "$@")
parse_dry_run() {
  for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true && break
  done
}
