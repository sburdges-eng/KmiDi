#!/usr/bin/env bash
set -e

OUT="_GIT_SAFE_CLEANUP_PLAN"
mkdir -p "$OUT"

echo "Generating git-safe cleanup plan..."
echo "Checking all branches for tracked files..."
echo

############################
# 1. Get all files tracked in ANY branch
############################
echo "Collecting files from all branches..."
ALL_GIT_FILES="$OUT/all_git_tracked_files.txt"
> "$ALL_GIT_FILES"

# Get unique files from all branches
for branch in $(git branch -a | grep -v HEAD | sed 's/remotes\/origin\///' | sed 's/^\*\? *//' | sort -u); do
  git ls-tree -r --name-only "$branch" 2>/dev/null >> "$ALL_GIT_FILES" || true
done

sort -u "$ALL_GIT_FILES" > "$ALL_GIT_FILES.tmp" && mv "$ALL_GIT_FILES.tmp" "$ALL_GIT_FILES"

GIT_FILE_COUNT=$(wc -l < "$ALL_GIT_FILES" | tr -d ' ')
echo "Found $GIT_FILE_COUNT unique files tracked across all branches"
echo

############################
# 2. Check what we already quarantined
############################
echo "Checking quarantined files against git..."
QUARANTINED_FILES="$OUT/quarantined_files_list.txt"
find _QUARANTINE_* -type f 2>/dev/null | sed "s|^_QUARANTINE_[^/]*/||" | sort -u > "$QUARANTINED_FILES" || touch "$QUARANTINED_FILES"

# Check if any quarantined files are in git
QUARANTINED_IN_GIT="$OUT/quarantined_files_in_git.txt"
comm -12 "$QUARANTINED_FILES" "$ALL_GIT_FILES" > "$QUARANTINED_IN_GIT" || touch "$QUARANTINED_IN_GIT"

QUARANTINED_GIT_COUNT=$(wc -l < "$QUARANTINED_IN_GIT" | tr -d ' ')
if [ "$QUARANTINED_GIT_COUNT" -gt 0 ]; then
  echo "⚠️  WARNING: $QUARANTINED_GIT_COUNT quarantined files are tracked in git!"
  echo "   Review: $QUARANTINED_IN_GIT"
else
  echo "✅ All quarantined files are safe (not in git)"
fi
echo

############################
# 3. Check aggressive cleanup candidates against git
############################
echo "Checking aggressive cleanup candidates..."
AGGRESSIVE_PLAN="_AGGRESSIVE_CLEANUP_PLAN"

# Check large trash directories
SAFE_TO_QUARANTINE="$OUT/safe_to_quarantine_dirs.txt"
UNSAFE_TO_QUARANTINE="$OUT/unsafe_to_quarantine_dirs.txt"
> "$SAFE_TO_QUARANTINE"
> "$UNSAFE_TO_QUARANTINE"

if [ -f "$AGGRESSIVE_PLAN/01_large_trash_directories.txt" ]; then
  while IFS= read -r dir; do
    if [ -d "$dir" ]; then
      # Check if any files in this directory are tracked in git
      dir_files=$(find "$dir" -type f 2>/dev/null | sed "s|^\./||")
      tracked_count=$(echo "$dir_files" | comm -12 - "$ALL_GIT_FILES" | wc -l | tr -d ' ')
      total_count=$(echo "$dir_files" | wc -l | tr -d ' ')
      
      if [ "$tracked_count" -eq 0 ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "$dir|$size|$total_count|0" >> "$SAFE_TO_QUARANTINE"
      else
        echo "$dir|$tracked_count tracked files|$total_count total" >> "$UNSAFE_TO_QUARANTINE"
      fi
    fi
  done < "$AGGRESSIVE_PLAN/01_large_trash_directories.txt"
fi

############################
# 4. Check cache directories
############################
CACHE_DIRS="$OUT/cache_dirs_git_check.txt"
> "$CACHE_DIRS"

if [ -f "$AGGRESSIVE_PLAN/02_all_cache_directories.txt" ]; then
  while IFS= read -r cache_dir; do
    if [ -d "$cache_dir" ]; then
      # Check if any files in cache dir are tracked
      cache_files=$(find "$cache_dir" -type f 2>/dev/null | sed "s|^\./||")
      tracked_in_cache=$(echo "$cache_files" | comm -12 - "$ALL_GIT_FILES" | wc -l | tr -d ' ')
      
      if [ "$tracked_in_cache" -gt 0 ]; then
        echo "$cache_dir|$tracked_in_cache tracked files" >> "$CACHE_DIRS"
      fi
    fi
  done < "$AGGRESSIVE_PLAN/02_all_cache_directories.txt"
fi

############################
# 5. Check potential backup directories
############################
BACKUP_DIRS_SAFE="$OUT/backup_dirs_safe.txt"
BACKUP_DIRS_UNSAFE="$OUT/backup_dirs_unsafe.txt"
> "$BACKUP_DIRS_SAFE"
> "$BACKUP_DIRS_UNSAFE"

if [ -f "$AGGRESSIVE_PLAN/04_potential_backup_directories.txt" ]; then
  while IFS= read -r dir; do
    if [ -d "$dir" ]; then
      dir_files=$(find "$dir" -type f 2>/dev/null | sed "s|^\./||")
      tracked_count=$(echo "$dir_files" | comm -12 - "$ALL_GIT_FILES" | wc -l | tr -d ' ')
      total_count=$(echo "$dir_files" | wc -l | tr -d ' ')
      
      if [ "$tracked_count" -eq 0 ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "$dir|$size|$total_count files" >> "$BACKUP_DIRS_SAFE"
      else
        echo "$dir|$tracked_count/$total_count files tracked in git" >> "$BACKUP_DIRS_UNSAFE"
      fi
    fi
  done < "$AGGRESSIVE_PLAN/04_potential_backup_directories.txt"
fi

############################
# 6. Generate summary
############################
{
  echo "=== GIT-SAFE CLEANUP PLAN SUMMARY ==="
  echo
  echo "Git tracking status:"
  echo "  Total unique files tracked across all branches: $GIT_FILE_COUNT"
  echo "  Quarantined files in git: $QUARANTINED_GIT_COUNT"
  if [ "$QUARANTINED_GIT_COUNT" -gt 0 ]; then
    echo "  ⚠️  WARNING: Some quarantined files are tracked in git!"
  else
    echo "  ✅ All quarantined files are safe"
  fi
  echo
  echo "Safe to quarantine (no git-tracked files):"
  safe_count=$(wc -l < "$SAFE_TO_QUARANTINE" | tr -d ' ')
  echo "  Directories: $safe_count"
  if [ "$safe_count" -gt 0 ]; then
    echo
    while IFS='|' read -r dir size files tracked; do
      printf "    %-30s %8s  %6s files\n" "$dir" "$size" "$files"
    done < "$SAFE_TO_QUARANTINE"
  fi
  echo
  echo "⚠️  UNSAFE to quarantine (contains git-tracked files):"
  unsafe_count=$(wc -l < "$UNSAFE_TO_QUARANTINE" | tr -d ' ')
  echo "  Directories: $unsafe_count"
  if [ "$unsafe_count" -gt 0 ]; then
    echo
    cat "$UNSAFE_TO_QUARANTINE" | sed 's/^/    /'
  fi
  echo
  echo "Backup directories safe to quarantine:"
  backup_safe=$(wc -l < "$BACKUP_DIRS_SAFE" | tr -d ' ')
  echo "  Count: $backup_safe"
  if [ "$backup_safe" -gt 0 ]; then
    echo
    while IFS='|' read -r dir size files; do
      printf "    %-30s %8s  %s\n" "$dir" "$size" "$files"
    done < "$BACKUP_DIRS_SAFE"
  fi
  echo
  echo "Cache directories with tracked files:"
  cache_tracked=$(wc -l < "$CACHE_DIRS" | tr -d ' ')
  echo "  Count: $cache_tracked"
  if [ "$cache_tracked" -gt 0 ]; then
    echo "  ⚠️  Review these - cache dirs shouldn't have tracked files!"
    cat "$CACHE_DIRS" | sed 's/^/    /'
  else
    echo "  ✅ No cache directories contain tracked files"
  fi
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
echo
echo "Detailed reports saved to $OUT/"
