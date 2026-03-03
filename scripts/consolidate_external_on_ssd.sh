#!/usr/bin/env bash
# Consolidate all *EXTERNAL* dirs/files on the SSD into their non-EXTERNAL counterpart names.
# Merges into existing dirs with rsync --ignore-existing (no overwrite = dedupe by path).
# Run from repo root or pass volume path. DRY-RUN by default; use --execute to apply.
#
# Usage:
#   ./scripts/consolidate_external_on_ssd.sh                    # dry-run
#   ./scripts/consolidate_external_on_ssd.sh --execute           # apply changes
#   ./scripts/consolidate_external_on_ssd.sh "/Volumes/Sean's SSD" --execute
set -e
DEFAULT_VOL=/Volumes/Sean\'s\ SSD
VOLUME="${1:-$DEFAULT_VOL}"
if [[ "$1" == "--execute" ]]; then
  VOLUME="$DEFAULT_VOL"
  EXECUTE=1
elif [[ "$2" == "--execute" ]]; then
  EXECUTE=1
else
  EXECUTE=0
fi

if [[ ! -d "$VOLUME" ]]; then
  echo "Volume not found: $VOLUME" >&2
  exit 1
fi

cd "$VOLUME"
echo "Volume: $VOLUME"
if [[ $EXECUTE -eq 1 ]]; then echo "Mode: EXECUTE"; else echo "Mode: DRY-RUN (use --execute to apply)"; fi
echo

# File: EXTERNAL_BUILD_REPORT.md -> BUILD_REPORT.md
if [[ -f "EXTERNAL_BUILD_REPORT.md" ]]; then
  if [[ ! -f "BUILD_REPORT.md" ]]; then
    if [[ $EXECUTE -eq 1 ]]; then
      mv "EXTERNAL_BUILD_REPORT.md" "BUILD_REPORT.md"
      echo "[EXEC] mv EXTERNAL_BUILD_REPORT.md -> BUILD_REPORT.md"
    else
      echo "[DRY] would mv EXTERNAL_BUILD_REPORT.md -> BUILD_REPORT.md"
    fi
  else
    echo "[SKIP] EXTERNAL_BUILD_REPORT.md (BUILD_REPORT.md already exists)"
  fi
fi

# Dirs: *EXTERNAL* -> strip EXTERNAL
for ext_dir in *EXTERNAL*; do
  [[ -e "$ext_dir" ]] || continue
  [[ -d "$ext_dir" ]] || continue
  base="${ext_dir%EXTERNAL}"
  if [[ -z "$base" ]]; then
    echo "[SKIP] $ext_dir (no base name after stripping EXTERNAL)"
    continue
  fi
  if [[ -d "$base" ]]; then
    # Merge: rsync --ignore-existing (don't overwrite = dedupe), then remove source
    if [[ $EXECUTE -eq 1 ]]; then
      rsync -a --ignore-existing "$ext_dir"/ "$base/"
      echo "[EXEC] merged $ext_dir -> $base (rsync --ignore-existing)"
      rm -rf "$ext_dir"
      echo "[EXEC] removed $ext_dir"
    else
      echo "[DRY] would merge $ext_dir -> $base (rsync --ignore-existing), then remove $ext_dir"
    fi
  else
    # No counterpart: rename
    if [[ $EXECUTE -eq 1 ]]; then
      mv "$ext_dir" "$base"
      echo "[EXEC] mv $ext_dir -> $base"
    else
      echo "[DRY] would mv $ext_dir -> $base"
    fi
  fi
done

echo
echo 'Done. Re-run without --execute to see what would happen (dry-run).'
