#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <directory_to_quarantine>"
  echo "Example: $0 'filer fuckery'"
  exit 1
fi

TARGET_DIR="$1"
QUARANTINE="_QUARANTINE_$(date +%Y%m%d_%H%M%S)"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' does not exist"
  exit 1
fi

mkdir -p "$QUARANTINE"

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
echo "To restore: rm '$TARGET_DIR' && mv '$QUARANTINE_TARGET' '$TARGET_DIR'"
echo "To delete: rm '$TARGET_DIR' && rm -rf '$QUARANTINE_TARGET'"
