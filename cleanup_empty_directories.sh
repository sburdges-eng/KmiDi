#!/usr/bin/env bash
set -e

OUT="_EMPTY_DIRS_CLEANUP"
mkdir -p "$OUT"

echo "Finding empty directories (excluding git, quarantine, and audit dirs)..."
echo

# Find empty directories, excluding:
# - .git directories
# - Quarantine directories
# - Audit/report directories
# - Hidden directories starting with .
find . -type d -empty 2>/dev/null | \
  grep -v "^\./\.git" | \
  grep -v "^\./\." | \
  grep -v "_QUARANTINE" | \
  grep -v "_AUDIT_REPORT" | \
  grep -v "_DELETION_PLAN" | \
  grep -v "_AGGRESSIVE_CLEANUP_PLAN" | \
  grep -v "_GIT_SAFE" | \
  grep -v "_EMPTY_DIRS" | \
  sort > "$OUT/empty_directories.txt"

count=$(wc -l < "$OUT/empty_directories.txt" | tr -d ' ')
echo "Found $count empty directories"
echo

# Check which ones are safe (not in git)
echo "Checking which are safe to remove (not in git)..."
SAFE_TO_REMOVE="$OUT/safe_to_remove.txt"
UNSAFE_TO_REMOVE="$OUT/unsafe_to_remove.txt"
> "$SAFE_TO_REMOVE"
> "$UNSAFE_TO_REMOVE"

GIT_DIRS=$(git ls-files | xargs dirname 2>/dev/null | sort -u)

while IFS= read -r dir; do
  # Check if any parent directory is in git
  parent="$dir"
  in_git=false
  
  while [ "$parent" != "." ] && [ "$parent" != "" ]; do
    if echo "$GIT_DIRS" | grep -q "^$parent$"; then
      in_git=true
      break
    fi
    parent=$(dirname "$parent")
  done
  
  if [ "$in_git" = false ]; then
    echo "$dir" >> "$SAFE_TO_REMOVE"
  else
    echo "$dir" >> "$UNSAFE_TO_REMOVE"
  fi
done < "$OUT/empty_directories.txt"

safe_count=$(wc -l < "$SAFE_TO_REMOVE" | tr -d ' ')
unsafe_count=$(wc -l < "$UNSAFE_TO_REMOVE" | tr -d ' ')

echo "Safe to remove: $safe_count"
echo "Unsafe (in git path): $unsafe_count"
echo

# Generate summary
{
  echo "=== EMPTY DIRECTORY CLEANUP SUMMARY ==="
  echo
  echo "Total empty directories: $count"
  echo "Safe to remove: $safe_count"
  echo "Unsafe (in git path): $unsafe_count"
  echo
  echo "To remove safe empty directories:"
  echo "  while IFS= read -r dir; do rmdir \"\$dir\" 2>/dev/null || true; done < $OUT/safe_to_remove.txt"
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
echo
echo "Lists saved to $OUT/"
