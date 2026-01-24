#!/usr/bin/env bash
# Phase 1.1: Quarantine temporary files (*.pyc, *.pyo, *.log, *.tmp, *.bak, .DS_Store)
# Usage: ./quarantine_temp_files.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

parse_dry_run "$@"
init_quarantine

QUARANTINE_DIR="$QUARANTINE_BASE/temp_files"
MANIFEST="$QUARANTINE_BASE/manifest_temp_files.json"

if [[ "$DRY_RUN" != "true" ]]; then
  mkdir -p "$QUARANTINE_DIR"
fi

log "Phase 1.1: Quarantining temporary files..."

# Patterns to find
PATTERNS=(
  "*.pyc"
  "*.pyo"
  "*.log"
  "*.tmp"
  "*.temp"
  "*.bak"
  "*.orig"
  ".DS_Store"
)

count=0
total_size=0
manifest_entries=()

for root in "${ROOTS[@]}"; do
  [[ ! -d "$root" ]] && continue

  for pattern in "${PATTERNS[@]}"; do
    while IFS= read -r -d '' file; do
      # Exclude venv paths
      if echo "$file" | grep -qE '/\.?venv[^/]*/|/venv/|/\.venv/'; then
        continue
      fi

      rel_path="${file#$root/}"
      dest="$QUARANTINE_DIR/$root/$rel_path"

      if [[ "$DRY_RUN" != "true" ]]; then
        size=$(du -sk "$file" 2> /dev/null | cut -f1 || echo 0)
        total_size=$((total_size + size))
        manifest_entries+=("{\"src\":\"$file\",\"dest\":\"$dest\",\"size\":$size}")
      fi

      run_mv "$file" "$dest"
      count=$((count + 1))

      if ((count % 100 == 0)); then
        log "Processed $count files..."
      fi
    done < <(find "$root" -type f -name "$pattern" -print0 2> /dev/null)
  done
done

log "Quarantined $count temporary files"

if [[ "$DRY_RUN" != "true" ]] && ((${#manifest_entries[@]} > 0)); then
  echo "[" > "$MANIFEST"
  for i in "${!manifest_entries[@]}"; do
    echo -n "${manifest_entries[$i]}" >> "$MANIFEST"
    ((i < ${#manifest_entries[@]} - 1)) && echo "," >> "$MANIFEST" || echo "" >> "$MANIFEST"
  done
  echo "]" >> "$MANIFEST"
  log "Manifest saved to $MANIFEST"
  log "Total size: $((total_size / 1024)) MB"
fi

log "Phase 1.1 complete"
