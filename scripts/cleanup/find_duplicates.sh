#!/usr/bin/env bash
# Phase 3.2: Find duplicate files across workspaces (hash-based)
# Generates docs/cleanup/duplicates_report.md
# Usage: ./find_duplicates.sh [--dry-run] [--limit N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/Users/seanburdges/KmiDi-1"
REPORT="$PROJECT_ROOT/docs/cleanup/duplicates_report.md"

ROOTS=(
  "/Users/seanburdges/KmiDi-1"
  "/Users/seanburdges/KmiDi"
)

LIMIT=500
DRY_RUN=false

log() { echo "[duplicates] $*"; }

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --limit=*) LIMIT="${arg#--limit=}" ;;
  esac
done

mkdir -p "$(dirname "$REPORT")"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

{
  echo "# Duplicate File Report"
  echo ""
  echo "**Generated:** $(date)"
  echo ""
  echo "Phase 3.2 of Safe Cleanup Plan. Hash-based duplicate detection. **Requires manual review before deletion.**"
  echo ""
  echo "---"
  echo ""
} > "$REPORT"

if [[ "$DRY_RUN" == "true" ]]; then
  log "[DRY-RUN] Would scan roots for duplicates (limit $LIMIT files)"
  {
    echo "**Mode:** Dry run (no scan performed)"
    echo ""
  } >> "$REPORT"
  exit 0
fi

log "Building file list (limit $LIMIT)..."
find "${ROOTS[@]}" -type f 2> /dev/null | head -n "$LIMIT" > "$TMPDIR/files.txt" || true
n=$(wc -l < "$TMPDIR/files.txt" | tr -d " ")
log "Hashing $n files..."

while IFS= read -r f; do
  [[ -f "$f" ]] || continue
  h=$(shasum -a 256 "$f" 2> /dev/null | cut -d" " -f1) || continue
  echo "$h	$f"
done < "$TMPDIR/files.txt" > "$TMPDIR/hashed.txt"

log "Finding duplicate hashes..."
sort -k1,1 "$TMPDIR/hashed.txt" | cut -f1 | uniq -d > "$TMPDIR/dup_hashes.txt" 2> /dev/null || true
dup_hashes=$(wc -l < "$TMPDIR/dup_hashes.txt" 2> /dev/null | tr -d " " || echo "0")

{
  echo "## Summary"
  echo ""
  echo "| Metric | Value |"
  echo "|--------|-------|"
  echo "| Files scanned | $n |"
  echo "| Duplicate hash groups | $dup_hashes |"
  echo ""
  echo "## Duplicate groups"
  echo ""
} >> "$REPORT"

if [[ "$dup_hashes" -gt 0 ]]; then
  g=1
  while IFS= read -r hash; do
    [[ -z "$hash" ]] && continue
    echo "### Group $g (hash \`${hash:0:16}...\`)" >> "$REPORT"
    grep "^$hash	" "$TMPDIR/hashed.txt" | cut -f2- | while read -r p; do
      echo "- \`$p\`" >> "$REPORT"
    done
    echo "" >> "$REPORT"
    g=$((g + 1))
  done < "$TMPDIR/dup_hashes.txt"
else
  echo "No duplicates found in scanned set. Try \`--limit=5000\` or install \`fdupes\` for full scan." >> "$REPORT"
fi

{
  echo "---"
  echo ""
  echo "## Next steps"
  echo ""
  echo "1. **Manual review:** Confirm duplicates before deleting."
  echo "2. Keep one copy, remove others."
  echo "3. Re-run with \`--limit=5000\` for broader scan."
} >> "$REPORT"

log "Report written to $REPORT"
