#!/usr/bin/env bash
set -e

echo "=== Executing Safe Cleanup ==="
echo "This will clean verified safe directories"
echo

# Track space freed
BEFORE_TOTAL=$(du -sh ~/Library/Caches ~/Library/Application\ Support/com.apple.wallpaper 2>/dev/null | awk '{sum+=$1} END {print sum}' || echo "0")

############################
# 1. Clean System Caches
############################
echo "1. Cleaning System Caches..."
CACHE_SIZE_BEFORE=$(du -sh ~/Library/Caches 2>/dev/null | cut -f1 || echo "0")
if [ -d ~/Library/Caches ] && [ "$(ls -A ~/Library/Caches 2>/dev/null)" ]; then
  rm -rf ~/Library/Caches/*
  echo "   ✅ Cleaned: ~/Library/Caches ($CACHE_SIZE_BEFORE)"
else
  echo "   ℹ️  Already clean or doesn't exist"
fi
echo

############################
# 2. Clean Wallpaper Cache
############################
echo "2. Cleaning Wallpaper Cache..."
WALLPAPER_DIR="$HOME/Library/Application Support/com.apple.wallpaper"
WALLPAPER_SIZE_BEFORE=$(du -sh "$WALLPAPER_DIR" 2>/dev/null | cut -f1 || echo "0")
if [ -d "$WALLPAPER_DIR" ] && [ "$(ls -A "$WALLPAPER_DIR" 2>/dev/null)" ]; then
  rm -rf "$WALLPAPER_DIR"/*
  echo "   ✅ Cleaned: $WALLPAPER_DIR ($WALLPAPER_SIZE_BEFORE)"
else
  echo "   ℹ️  Already clean or doesn't exist"
fi
echo

############################
# 3. Show iOS Backup Status
############################
echo "3. iOS Backups Status..."
MOBILESYNC="$HOME/Library/Application Support/MobileSync/Backup"
if [ -d "$MOBILESYNC" ]; then
  BACKUP_COUNT=$(find "$MOBILESYNC" -maxdepth 1 -type d | wc -l | tr -d ' ')
  BACKUP_SIZE=$(du -sh "$MOBILESYNC" 2>/dev/null | cut -f1 || echo "0")
  echo "   📱 Backups found: $BACKUP_COUNT directories"
  echo "   💾 Total size: $BACKUP_SIZE"
  echo "   ℹ️  Review manually: ls -lh $MOBILESYNC"
  echo "   ℹ️  Remove old backups as needed (keep only recent ones)"
else
  echo "   ℹ️  No backups found"
fi
echo

############################
# 4. Summary
############################
AFTER_CACHE=$(du -sh ~/Library/Caches 2>/dev/null | cut -f1 || echo "0")
AFTER_WALLPAPER=$(du -sh "$WALLPAPER_DIR" 2>/dev/null | cut -f1 || echo "0")

echo "=== Cleanup Complete ==="
echo
echo "Caches before: $CACHE_SIZE_BEFORE"
echo "Caches after:  $AFTER_CACHE"
echo
echo "Wallpapers before: $WALLPAPER_SIZE_BEFORE"
echo "Wallpapers after:  $AFTER_WALLPAPER"
echo
echo "✅ Safe cleanup complete!"
echo
echo "Next steps:"
echo "  - Review iOS backups in: $MOBILESYNC"
echo "  - Review iCloud Desktop/Documents sync (98GB)"
echo "  - Review OneDrive settings (68GB)"
