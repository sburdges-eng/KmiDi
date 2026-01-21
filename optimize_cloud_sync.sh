#!/usr/bin/env bash
set -e

echo "=== Cloud Sync Optimization Guide ==="
echo

############################
# 1. iCloud Desktop/Documents
############################
echo "1. iCloud Desktop/Documents Sync (98GB)"
echo "   Location: ~/Library/Application Support/CloudDocs"
echo
CLOUDDOCS_SIZE=$(du -sh ~/Library/Application\ Support/CloudDocs 2>/dev/null | cut -f1 || echo "Unknown")
echo "   Current size: $CLOUDDOCS_SIZE"
echo
echo "   To disable sync (files stay local):"
echo "   1. Open System Settings"
echo "   2. Click your Apple ID (top)"
echo "   3. Click 'iCloud'"
echo "   4. Click 'iCloud Drive'"
echo "   5. Turn OFF 'Desktop & Documents Folders'"
echo
echo "   ⚠️  Note: Files remain on your Mac, just stop syncing to iCloud"
echo "   💾 This will free up the 98GB in CloudDocs (local files unaffected)"
echo
echo

############################
# 2. OneDrive
############################
echo "2. OneDrive Cache (68GB)"
echo "   Location: ~/Library/Group Containers/UBF8T346G9.OneDriveStandaloneSuite"
echo
ONEDRIVE_SIZE=$(du -sh ~/Library/Group\ Containers/UBF8T346G9.OneDriveStandaloneSuite 2>/dev/null | cut -f1 || echo "Unknown")
echo "   Current size: $ONEDRIVE_SIZE"
echo
echo "   To optimize:"
echo "   1. Open OneDrive app"
echo "   2. Click OneDrive icon in menu bar"
echo "   3. Click Settings (gear icon)"
echo "   4. Go to 'Account' tab"
echo "   5. Enable 'Files On-Demand'"
echo "      - Files download only when you access them"
echo "      - Reduces local cache size significantly"
echo
echo "   Alternative: Review which folders are syncing"
echo "   - Uncheck folders you don't need locally"
echo "   - They'll still be available in OneDrive web interface"
echo
echo

############################
# 3. Summary
############################
echo "=== Expected Savings ==="
echo
echo "iCloud Desktop/Documents:"
echo "  - Disabling sync: ~98GB freed from CloudDocs"
echo "  - Files remain local (no data loss)"
echo
echo "OneDrive:"
echo "  - Enabling Files On-Demand: ~20-40GB freed"
echo "  - Selective sync: Additional savings possible"
echo
echo "Total potential: ~118-138GB"
echo
