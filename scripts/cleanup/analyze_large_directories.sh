#!/usr/bin/env bash
# Phase 3.1: Analyze large directories (RECOVERY_OPS, Desktop, audio)
# Generates docs/cleanup/large_dirs_analysis.md
# Usage: ./analyze_large_directories.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/Users/seanburdges/KmiDi-1"
REPORT="$PROJECT_ROOT/docs/cleanup/large_dirs_analysis.md"

LARGE_DIRS=(
  "/Users/seanburdges/RECOVERY_OPS"
  "/Users/seanburdges/Desktop"
  "/Users/seanburdges/audio"
)

log() { echo "[analyze] $*"; }

mkdir -p "$(dirname "$REPORT")"

{
  echo "# Large Directory Analysis"
  echo ""
  echo "**Generated:** $(date)"
  echo ""
  echo "Phase 3.1 of Safe Cleanup Plan. Identifies cleanup opportunities in large workspace directories."
  echo ""
  echo "---"
  echo ""
} > "$REPORT"

for dir in "${LARGE_DIRS[@]}"; do
  if [[ ! -d "$dir" ]]; then
    log "Skipping $dir (not found)"
    {
      echo "## $(basename "$dir")"
      echo ""
      echo "**Status:** Directory not found or not accessible"
      echo ""
    } >> "$REPORT"
    continue
  fi

  log "Analyzing $dir..."
  size=$(du -sh "$dir" 2> /dev/null | head -1 | awk '{print $1}' || echo "unknown")

  {
    echo "## $(basename "$dir")"
    echo ""
    echo "| Metric | Value |"
    echo "|--------|-------|"
    echo "| Path | \`$dir\` |"
    echo "| Total size | $size |"
    echo ""
  } >> "$REPORT"

  {
    echo "### Largest subdirectories"
    echo ""
    echo "| Size | Path |"
    echo "|------|------|"
    subdirs=""
    if [[ "$(basename "$dir")" == "RECOVERY_OPS" ]]; then
      subdirs=""
      echo "| (skipped for very large tree) | |"
    else
      subdirs=$(cd "$dir" 2> /dev/null && for d in */; do [[ -d "$d" ]] && du -sh "$d" 2> /dev/null; done | sort -hr | head -15 || true)
      echo "$subdirs" | while read -r s p; do [[ -n "$s" ]] && echo "| $s | \`${p%/}\` |"; done
    fi
    echo ""
  } >> "$REPORT"

  {
    echo "### Cleanup opportunities"
    echo ""
    case "$(basename "$dir")" in
      RECOVERY_OPS)
        echo "- **Archive review:** Identify recovery/backup content that can be archived or removed."
        echo "- **Duplicate detection:** Run \`find_duplicates.sh\` in this tree."
        echo "- **Retention policy:** Consider age-based cleanup of old recovery items."
        ;;
      Desktop)
        echo "- **Temporary files:** Identify old downloads, screenshots, or temp files."
        echo "- **Project clutter:** Move completed project folders to archive or Documents."
        echo "- ***.dmg, *.pkg:** Remove installers after use."
        ;;
      audio)
        echo "- **Duplicate audio:** Run \`find_duplicates.sh\` for duplicate tracks."
        echo "- **Unused stems/sessions:** Archive or remove old project audio."
        echo "- **Format normalization:** Consider deduplicating same content in different formats."
        ;;
      *)
        echo "- Run \`find_duplicates.sh\` for duplicate detection."
        echo "- Manual review recommended."
        ;;
    esac
    echo ""
  } >> "$REPORT"
done

{
  echo "---"
  echo ""
  echo "## Next steps"
  echo ""
  echo "1. Run \`find_duplicates.sh\` for duplicate file detection across workspaces."
  echo "2. Manually review directories before deleting or archiving."
  echo "3. Use quarantine scripts (Phase 1–2) only for safe, regenerable artifacts."
} >> "$REPORT"

log "Report written to $REPORT"
