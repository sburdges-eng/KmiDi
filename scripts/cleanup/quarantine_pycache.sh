#!/usr/bin/env bash
# Phase 1.3: Quarantine Python cache directories, tool caches, and .DS_Store files
# Usage: ./quarantine_pycache.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

parse_dry_run "$@"
init_quarantine

QUARANTINE_DIR="$QUARANTINE_BASE/pycache"
MANIFEST="$QUARANTINE_BASE/manifest_pycache.json"

if [[ "$DRY_RUN" != "true" ]]; then
  mkdir -p "$QUARANTINE_DIR"
fi

log "Phase 1.3: Quarantining Python cache and tool caches..."

# Directories to quarantine
CACHE_DIRS=(
  "__pycache__"
  ".mypy_cache"
  ".pytest_cache"
  ".ruff_cache"
  ".coverage"
  "htmlcov"
  ".hypothesis"
)

count=0
total_size=0
manifest_entries=()

for root in "${ROOTS[@]}"; do
  [[ ! -d "$root" ]] && continue

  for cache_dir in "${CACHE_DIRS[@]}"; do
    while IFS= read -r -d '' dir; do
      # Exclude venv paths
      if echo "$dir" | grep -qE '/\.?venv[^/]*/|/venv/|/\.venv/'; then
        continue
      fi

      # Exclude quarantine directories themselves
      if echo "$dir" | grep -qE '_QUARANTINE_'; then
        continue
      fi

      rel_path="${dir#$root/}"
      dest="$QUARANTINE_DIR/$root/$rel_path"

      if [[ "$DRY_RUN" != "true" ]]; then
        size=$(du -sk "$dir" 2> /dev/null | cut -f1 || echo 0)
        total_size=$((total_size + size))
        manifest_entries+=("{\"src\":\"$dir\",\"dest\":\"$dest\",\"size\":$size}")
      fi

      run_mv "$dir" "$dest"
      count=$((count + 1))

      if ((count % 50 == 0)); then
        log "Processed $count cache directories..."
      fi
    done < <(find "$root" -type d -name "$cache_dir" -print0 2> /dev/null)
  done
done

log "Quarantined $count cache directories"

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

log "Phase 1.3 complete"
