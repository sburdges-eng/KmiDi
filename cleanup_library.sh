#!/usr/bin/env bash
set -e

echo "=== LIBRARY CLEANUP ANALYSIS ==="
echo
echo "Found the following large directories:"
echo

echo "1. CloudDocs (iCloud Desktop/Documents): 98GB"
echo "   Location: ~/Library/Application Support/CloudDocs"
echo "   Action: Review iCloud Desktop/Documents sync settings"
echo "   Safe to clean: NO (synced files)"
echo

echo "2. OneDrive Standalone Suite: 68GB"
echo "   Location: ~/Library/Group Containers/UBF8T346G9.OneDriveStandaloneSuite"
echo "   Action: Review OneDrive cache/sync"
echo "   Safe to clean: PARTIAL (cache may be safe)"
echo

echo "3. MobileSync (iOS backups): 23GB"
echo "   Location: ~/Library/Application Support/MobileSync"
echo "   Action: Review old iOS backups"
echo "   Safe to clean: YES (old backups can be deleted)"
echo

echo "4. com.apple.wallpaper: 11GB"
echo "   Location: ~/Library/Application Support/com.apple.wallpaper"
echo "   Action: Review if you need all wallpapers"
echo "   Safe to clean: YES (regenerates)"
echo

echo "5. Caches: 11GB"
echo "   Location: ~/Library/Caches"
echo "   Action: Can safely clean"
echo "   Safe to clean: YES"
echo

echo "=== RECOMMENDED ACTIONS ==="
echo
echo "SAFE (can do now):"
echo "  1. Clean caches: rm -rf ~/Library/Caches/*"
echo "  2. Review old iOS backups in MobileSync"
echo "  3. Review wallpaper files (if not needed)"
echo
echo "REVIEW FIRST:"
echo "  1. Check iCloud Desktop/Documents sync (98GB)"
echo "     - Settings > Apple ID > iCloud > iCloud Drive"
echo "     - Consider disabling Desktop/Documents sync if not needed"
echo
echo "  2. Check OneDrive settings (68GB)"
echo "     - Review what's being synced"
echo     - Check cache size in OneDrive preferences
echo
echo "=== QUICK WINS ==="
echo
echo "Immediate cleanup (safe, saves ~22GB):"
echo "  rm -rf ~/Library/Caches/*"
echo "  # Review and remove old iOS backups:"
echo "  ls -lh ~/Library/Application\ Support/MobileSync/Backup/"
echo
echo "Potential total savings: ~200GB+"
