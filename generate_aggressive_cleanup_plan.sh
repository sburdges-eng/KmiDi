#!/usr/bin/env bash
set -e

OUT="_AGGRESSIVE_CLEANUP_PLAN"
AUDIT_DIR="$(ls -d _AUDIT_REPORT_* | tail -1)"

mkdir -p "$OUT"

echo "Generating aggressive cleanup plan..."
echo "Using audit: $AUDIT_DIR"
echo

############################
# 1. Large obvious trash directories
############################
cat <<EOF > "$OUT/01_large_trash_directories.txt"
filer fuckery
.mypy_cache
__pycache__
.pytest_cache
.cache
EOF

############################
# 2. Cache and build directories (all instances)
############################
{
  find . -type d -name "__pycache__" 2>/dev/null
  find . -type d -name ".mypy_cache" 2>/dev/null
  find . -type d -name ".pytest_cache" 2>/dev/null
  find . -type d -name ".cache" 2>/dev/null
  find . -type d -name "node_modules" 2>/dev/null
  find . -type d -name ".venv" 2>/dev/null
  find . -type d -name "venv" 2>/dev/null
} | sort -u > "$OUT/02_all_cache_directories.txt"

############################
# 3. Empty directories
############################
find . -type d -empty 2>/dev/null | sort > "$OUT/03_empty_directories.txt"

############################
# 4. Large duplicate/backup directories
############################
cat <<EOF > "$OUT/04_potential_backup_directories.txt"
KmiDi
kelly-project
ARCHIVE
CONSOLIDATED_CODE
Stashed_Changes
midikompanion-claude-upload
EOF

############################
# 5. Summary with sizes
############################
{
  echo "=== AGGRESSIVE CLEANUP PLAN SUMMARY ==="
  echo
  echo "Large trash directories (should quarantine entire dirs):"
  while IFS= read -r dir; do
    if [ -d "$dir" ]; then
      size=$(du -sh "$dir" 2>/dev/null | cut -f1)
      count=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
      printf "  %-30s %8s  %6d files\n" "$dir" "$size" "$count"
    fi
  done < "$OUT/01_large_trash_directories.txt"
  echo
  echo "Cache directories found:"
  wc -l < "$OUT/02_all_cache_directories.txt"
  echo
  echo "Empty directories:"
  wc -l < "$OUT/03_empty_directories.txt"
  echo
  echo "Potential backup/duplicate directories:"
  while IFS= read -r dir; do
    if [ -d "$dir" ]; then
      size=$(du -sh "$dir" 2>/dev/null | cut -f1)
      count=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
      printf "  %-30s %8s  %6d files\n" "$dir" "$size" "$count"
    fi
  done < "$OUT/04_potential_backup_directories.txt"
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
echo
echo "Detailed plans saved to $OUT/"
echo "REVIEW CAREFULLY before using quarantine_directory.sh"
