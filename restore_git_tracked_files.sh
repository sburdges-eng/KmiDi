#!/usr/bin/env bash
set -e

GIT_TRACKED_LIST="_GIT_SAFE_CLEANUP_PLAN/quarantined_files_in_git.txt"
QUARANTINE_BASE="_QUARANTINE_20260119_042228"

if [ ! -f "$GIT_TRACKED_LIST" ]; then
  echo "Error: Git tracked files list not found"
  echo "Run generate_git_safe_cleanup_plan.sh first"
  exit 1
fi

echo "Restoring git-tracked files from quarantine..."
echo "This will restore files that are tracked in git but were quarantined"
echo

# Find which quarantine directory contains each file
restored=0
skipped=0

while IFS= read -r file; do
  # Check all quarantine directories
  found=false
  for quarantine in _QUARANTINE_*; do
    if [ -f "$quarantine/$file" ]; then
      # Create parent directory if needed
      mkdir -p "$(dirname "$file")"
      # Restore the file
      cp "$quarantine/$file" "$file"
      echo "Restored: $file"
      restored=$((restored + 1))
      found=true
      break
    fi
  done
  
  if [ "$found" = false ]; then
    # File might be in a directory that was moved as a whole
    # Check if parent directory is a symlink
    parent=$(dirname "$file")
    if [ -L "$parent" ] || [ -L "$(dirname "$parent")" ]; then
      echo "Skipped (accessible via symlink): $file"
      skipped=$((skipped + 1))
    else
      echo "Warning: Could not find in quarantine: $file"
    fi
  fi
done < "$GIT_TRACKED_LIST"

echo
echo "Restored: $restored files"
echo "Skipped (accessible via symlink): $skipped files"
echo
echo "Note: Files in 'filer fuckery' and '.mypy_cache' are accessible via symlinks"
echo "If you need to restore those, remove the symlink and move the directory back"
