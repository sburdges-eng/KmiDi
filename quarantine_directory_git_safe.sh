#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <directory_to_quarantine>"
  echo "Example: $0 'filer fuckery'"
  exit 1
fi

TARGET_DIR="$1"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' does not exist"
  exit 1
fi

# Check if directory contains git-tracked files
echo "Checking for git-tracked files in '$TARGET_DIR'..."
GIT_TRACKED=$(git ls-files "$TARGET_DIR" 2>/dev/null | wc -l | tr -d ' ')

if [ "$GIT_TRACKED" -gt 0 ]; then
  echo "⚠️  WARNING: Directory contains $GIT_TRACKED git-tracked files!"
  echo
  echo "Tracked files:"
  git ls-files "$TARGET_DIR" | head -10
  if [ "$GIT_TRACKED" -gt 10 ]; then
    echo "... and $((GIT_TRACKED - 10)) more"
  fi
  echo
  read -p "Continue anyway? (yes/no): " confirm
  if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
  fi
fi

QUARANTINE="_QUARANTINE_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$QUARANTINE"

echo
echo "Quarantining directory: $TARGET_DIR"
echo "Into: $QUARANTINE"
echo

# Get absolute path and move
ABS_TARGET=$(cd "$(dirname "$TARGET_DIR")" && pwd)/$(basename "$TARGET_DIR")
QUARANTINE_TARGET="$QUARANTINE/$(basename "$TARGET_DIR")"

mv "$TARGET_DIR" "$QUARANTINE_TARGET"

# Create symlink so nothing breaks
ln -s "$(pwd)/$QUARANTINE_TARGET" "$TARGET_DIR"

echo
echo "Directory quarantined: $QUARANTINE_TARGET"
echo "Symlink created at: $TARGET_DIR -> $QUARANTINE_TARGET"
if [ "$GIT_TRACKED" -gt 0 ]; then
  echo "⚠️  Note: $GIT_TRACKED git-tracked files are now in quarantine"
  echo "   Files are accessible via symlink, but consider updating .gitignore"
fi
echo
echo "To restore: rm '$TARGET_DIR' && mv '$QUARANTINE_TARGET' '$TARGET_DIR'"
echo "To delete: rm '$TARGET_DIR' && rm -rf '$QUARANTINE_TARGET'"
