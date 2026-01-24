#!/usr/bin/env bash
# Phase 1.2: Remove empty directories
# Usage: ./quarantine_empty_dirs.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

parse_dry_run "$@"
init_quarantine

EMPTY_DIRS_LOG="$QUARANTINE_BASE/empty_dirs.txt"

log "Phase 1.2: Removing empty directories..."

count=0

for root in "${ROOTS[@]}"; do
  [[ ! -d "$root" ]] && continue

  # Find empty directories, excluding venvs
  while IFS= read -r -d '' dir; do
    # Exclude venv paths
    if echo "$dir" | grep -qE '/\.?venv[^/]*/|/venv/|/\.venv/'; then
      continue
    fi

    # Exclude quarantine directories themselves
    if echo "$dir" | grep -qE '_QUARANTINE_'; then
      continue
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
      echo "$dir" >> "$EMPTY_DIRS_LOG"
    else
      log "[DRY-RUN] Would remove: $dir"
    fi

    run_rmdir "$dir"
    count=$((count + 1))

    if ((count % 1000 == 0)); then
      log "Processed $count empty directories..."
    fi
  done < <(find "$root" -type d -empty -print0 2> /dev/null)
done

log "Removed $count empty directories"
log "List saved to $EMPTY_DIRS_LOG"
log "Phase 1.2 complete"
