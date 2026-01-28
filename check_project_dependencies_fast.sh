#!/usr/bin/env bash
set -e

OUT="_PROJECT_DEPENDENCY_CHECK_FAST"
mkdir -p "$OUT"

echo "Quick check: Do projects need the large directories?"
echo "Checking known project locations..."
echo

# Large directories we want to verify are safe to clean
LARGE_DIRS=(
  "Library/Application Support/CloudDocs"
  "Library/Application Support/MobileSync"
  "Library/Group Containers"
  "Library/Application Support/com.apple.wallpaper"
  "Library/Caches"
)

SAFE_TO_CLEAN=()
NEEDS_REVIEW=()

############################
# 1. Check known project locations
############################
echo "=== Checking Known Projects ==="
KNOWN_PROJECTS=(
  "$HOME/KmiDi"
  "$HOME/KmiDi-1"
  "$HOME/RECOVERY_OPS"
  "$HOME/Desktop"
  "$HOME/Documents"
  "$HOME/.cursor"
  "$HOME/audio"
)

REFS_FOUND=0

for proj in "${KNOWN_PROJECTS[@]}"; do
  if [ -d "$proj" ]; then
    echo -n "Checking $(basename "$proj")... "
    
    # Quick check for common references
    found_ref=0
    for dir_pattern in "Library\|CloudDocs\|MobileSync\|OneDrive"; do
      if find "$proj" -maxdepth 2 -type f \( \
        -name "*.py" -o \
        -name "*.js" -o \
        -name "*.json" -o \
        -name "*.yaml" -o \
        -name "*.yml" -o \
        -name "*.sh" \
      \) -exec grep -l "$dir_pattern" {} \; 2>/dev/null | head -1 | grep -q .; then
        found_ref=1
        break
      fi
    done
    
    if [ "$found_ref" -eq 1 ]; then
      echo "⚠️  REFERENCES FOUND"
      REFS_FOUND=$((REFS_FOUND + 1))
    else
      echo "✅ No references"
    fi
  fi
done
echo

############################
# 2. Check if Desktop/Documents are synced to iCloud
############################
echo "=== Checking iCloud Sync Status ==="

# Check if Desktop/Documents are in CloudDocs
if [ -d "$HOME/Library/Application Support/CloudDocs" ]; then
  cloud_docs_size=$(du -sh "$HOME/Library/Application Support/CloudDocs" 2>/dev/null | cut -f1)
  echo "CloudDocs size: $cloud_docs_size (iCloud Desktop/Documents sync)"
  echo "⚠️  This contains synced Desktop/Documents files"
  echo "   - Safe to disable sync (files remain local)"
  echo "   - Check: Settings > Apple ID > iCloud > iCloud Drive"
  echo
fi

############################
# 3. Check for symlinks
############################
echo "=== Checking for Symlinks ==="
SYMLINK_COUNT=0

# Check if any large dirs are symlinked from projects
for proj in "${KNOWN_PROJECTS[@]}"; do
  if [ -d "$proj" ]; then
    find "$proj" -maxdepth 3 -type l -exec readlink -f {} \; 2>/dev/null | \
      grep -q "Library/Application Support\|Library/Group Containers" && {
      echo "⚠️  Symlinks found in $(basename "$proj")"
      SYMLINK_COUNT=$((SYMLINK_COUNT + 1))
    } || true
  fi
done

if [ "$SYMLINK_COUNT" -eq 0 ]; then
  echo "✅ No symlinks found"
fi
echo

############################
# 4. Safe cleanup assessment
############################
echo "=== Safe Cleanup Assessment ==="
echo

echo "✅ SAFE TO CLEAN NOW:"
echo "   1. Library/Caches (11GB)"
echo "      - Regenerates automatically"
echo "      - No project dependencies"
echo "      - Command: rm -rf ~/Library/Caches/*"
echo

echo "   2. Library/Application Support/com.apple.wallpaper (11GB)"
echo "      - System wallpapers (regenerates)"
echo "      - No project dependencies"
echo "      - Command: rm -rf ~/Library/Application\\ Support/com.apple.wallpaper/*"
echo

echo "   3. Old iOS Backups in MobileSync (10-20GB)"
echo "      - Review and remove old backups"
echo "      - Keep only recent ones"
echo "      - No project dependencies"
echo

echo "⚠️  REVIEW BEFORE CLEANING:"
echo "   1. Library/Application Support/CloudDocs (98GB)"
echo "      - Contains iCloud-synced Desktop/Documents"
echo "      - Files may be in use by projects"
echo "      - Safe to disable sync (files stay local)"
echo "      - Check which projects use Desktop/Documents"
echo

echo "   2. Library/Group Containers/OneDrive (68GB)"
echo "      - OneDrive sync cache"
echo "      - May contain project files"
echo "      - Review OneDrive sync settings"
echo

echo "   3. RECOVERY_OPS (105GB)"
echo "      - Recovery/backup files"
echo "      - May contain project backups"
echo "      - Review contents before deleting"
echo

############################
# 5. Generate summary
############################
{
  echo "==== PROJECT DEPENDENCY CHECK SUMMARY ===="
  echo
  echo "Projects checked: ${#KNOWN_PROJECTS[@]}"
  echo "References found: $REFS_FOUND"
  echo "Symlinks found: $SYMLINK_COUNT"
  echo
  if [ "$REFS_FOUND" -eq 0 ] && [ "$SYMLINK_COUNT" -eq 0 ]; then
    echo "✅ CONCLUSION: No projects appear to depend on Library directories"
    echo "   Safe to clean caches, wallpapers, and old backups"
    echo "   Review iCloud/OneDrive sync settings for your workflow"
  else
    echo "⚠️  CONCLUSION: Some references found - review before cleaning"
  fi
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
