#!/usr/bin/env bash
set -e

BACKUP_DIR="$HOME/Library/Application Support/MobileSync/Backup"

if [ ! -d "$BACKUP_DIR" ]; then
  echo "No iOS backups found"
  exit 0
fi

echo "=== iOS Backup Cleanup ==="
echo
echo "Backup directory: $BACKUP_DIR"
echo

# List all backups with details
echo "Current backups:"
echo "---"
cd "$BACKUP_DIR"
for backup in */; do
  if [ -d "$backup" ]; then
    backup_name=$(basename "$backup")
    size=$(du -sh "$backup" 2>/dev/null | cut -f1)
    # Try to get device info from Info.plist
    if [ -f "$backup/Info.plist" ]; then
      device_name=$(plutil -extract "Display Name" raw "$backup/Info.plist" 2>/dev/null || echo "Unknown")
      backup_date=$(plutil -extract "Last Backup Date" raw "$backup/Info.plist" 2>/dev/null || echo "Unknown")
      echo "  $backup_name"
      echo "    Device: $device_name"
      echo "    Date: $backup_date"
      echo "    Size: $size"
      echo
    else
      echo "  $backup_name - $size"
    fi
  fi
done

echo "---"
echo
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
echo "Total backup size: $TOTAL_SIZE"
echo

echo "Recommendations:"
echo "  1. Keep only the most recent backup for each device"
echo "  2. Older backups can be safely removed"
echo "  3. Backups are stored in iCloud if you have iCloud Backup enabled"
echo
echo "To remove a backup:"
echo "  rm -rf \"$BACKUP_DIR/BACKUP_UDID\""
echo
echo "⚠️  Be careful - only remove backups you're sure you don't need!"
