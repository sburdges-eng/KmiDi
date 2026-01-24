#!/bin/bash
set -e

# Quarantine Mover
# Safely moves files to a quarantine directory (preserves directory structure)

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

PLAN_DIR="_DELETION_PLAN"

if [ ! -f "$PLAN_DIR/01_likely_trash_candidates.txt" ]; then
    echo "Error: Deletion plan not found. Please run generate_deletion_plan.sh first."
    exit 1
fi

# Create timestamped quarantine directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
QUARANTINE_DIR="_QUARANTINE_${TIMESTAMP}"
mkdir -p "$QUARANTINE_DIR"

echo "Quarantine directory: $QUARANTINE_DIR"
echo ""
echo "This script will move files from the deletion plan to quarantine."
echo "Files will be moved (not copied) to preserve directory structure."
echo ""

# Confirm
read -p "Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

MOVED_COUNT=0
SKIPPED_COUNT=0

# Process likely trash candidates
echo "Moving likely trash candidates..."
while IFS= read -r filepath || [ -n "$filepath" ]; do
    # Skip empty lines and comments
    [[ -z "$filepath" ]] && continue
    [[ "$filepath" =~ ^# ]] && continue
    
    # Remove leading ./ if present
    filepath="${filepath#./}"
    
    if [ ! -f "$filepath" ] && [ ! -d "$filepath" ]; then
        echo "  Skipping (not found): $filepath"
        ((SKIPPED_COUNT++))
        continue
    fi
    
    # Create target directory preserving structure
    target_dir="$QUARANTINE_DIR/$(dirname "$filepath")"
    mkdir -p "$target_dir"
    
    # Move file/directory
    if mv "$filepath" "$QUARANTINE_DIR/$filepath" 2>/dev/null; then
        echo "  Moved: $filepath"
        ((MOVED_COUNT++))
    else
        echo "  Failed to move: $filepath"
        ((SKIPPED_COUNT++))
    fi
done < "$PLAN_DIR/01_likely_trash_candidates.txt"

# Create manifest
{
    echo "Quarantine Manifest"
    echo "==================="
    echo "Created: $(date)"
    echo "Moved files: $MOVED_COUNT"
    echo "Skipped files: $SKIPPED_COUNT"
    echo ""
    echo "To restore files, run:"
    echo "  mv $QUARANTINE_DIR/* ."
    echo ""
    echo "Or restore individual files/directories from $QUARANTINE_DIR/"
} > "$QUARANTINE_DIR/MANIFEST.txt"

echo ""
echo "Quarantine complete!"
echo "  Moved: $MOVED_COUNT files"
echo "  Skipped: $SKIPPED_COUNT files"
echo ""
echo "Review your project and run tests/builds."
echo "If everything works, you can delete: $QUARANTINE_DIR"
echo "If something broke, restore files from: $QUARANTINE_DIR"
