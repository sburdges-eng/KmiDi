#!/bin/bash
set -e

# Deletion Plan Generator
# Reads the most recent audit report and generates a deletion plan

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Find the most recent audit report
LATEST_AUDIT=$(ls -td _AUDIT_REPORT_* 2>/dev/null | head -1)

if [ -z "$LATEST_AUDIT" ]; then
    echo "Error: No audit report found. Please run one_shot_audit.sh first."
    exit 1
fi

echo "Using audit report: $LATEST_AUDIT"

# Create deletion plan directory
PLAN_DIR="_DELETION_PLAN"
mkdir -p "$PLAN_DIR"

# Source file extensions to exclude from large file analysis
SOURCE_EXTS="\.py$|\.cpp$|\.h$|\.hpp$|\.c$|\.rs$|\.js$|\.ts$|\.md$|\.txt$|\.json$|\.yaml$|\.yml$|\.tsx$|\.jsx$|\.java$|\.kt$|\.swift$|\.go$|\.rb$|\.php$"

# 1. Copy likely trash candidates
echo "Generating likely trash candidates list..."
if [ -f "$LATEST_AUDIT/likely_trash_files.txt" ]; then
    grep -v "^Likely trash" "$LATEST_AUDIT/likely_trash_files.txt" | \
        grep -v "^===" | \
        grep -v "^$" > "$PLAN_DIR/01_likely_trash_candidates.txt" || true
    TRASH_COUNT=$(wc -l < "$PLAN_DIR/01_likely_trash_candidates.txt" | tr -d ' ')
    echo "  Found $TRASH_COUNT trash candidates"
else
    touch "$PLAN_DIR/01_likely_trash_candidates.txt"
    echo "  No trash files found"
fi

# 2. Large non-source files (top 500)
echo "Identifying large non-source files..."
{
    find . -type f -exec ls -lh {} \; 2>/dev/null | \
        awk '{print $5, $9}' | \
        while read size path; do
            # Skip source files
            if echo "$path" | grep -qiE "$SOURCE_EXTS"; then
                continue
            fi
            # Convert human-readable size to bytes for sorting (simplified approach)
            echo "$size $path"
        done | \
        sort -hr | \
        head -500
} > "$PLAN_DIR/02_large_non_source_files.txt"
LARGE_COUNT=$(wc -l < "$PLAN_DIR/02_large_non_source_files.txt" | tr -d ' ')
echo "  Found $LARGE_COUNT large non-source files"

# 3. Duplicate files
echo "Processing duplicate file groups..."
if [ -f "$LATEST_AUDIT/duplicate_hashes.txt" ]; then
    {
        echo "# Duplicate file groups"
        echo "# Keep the first file in each group, consider deleting the rest"
        echo ""
        grep -A 100 "^===" "$LATEST_AUDIT/duplicate_hashes.txt" | \
            awk '
                /^=== Hash:/ { 
                    if (group_count > 0) print ""; 
                    group_count++; 
                    print $0; 
                    next 
                }
                /^  \./ { 
                    files[++file_idx] = $0 
                }
                END {
                    for (i=2; i<=file_idx; i++) {
                        print files[i] " # DUPLICATE - consider deleting"
                    }
                }
            '
    } > "$PLAN_DIR/03_duplicate_files.txt" || true
    DUPLICATE_COUNT=$(grep -c "# DUPLICATE" "$PLAN_DIR/03_duplicate_files.txt" 2>/dev/null || echo "0")
    echo "  Found $DUPLICATE_COUNT duplicate file candidates"
else
    touch "$PLAN_DIR/03_duplicate_files.txt"
    echo "  No duplicates found"
fi

# 4. Generate summary
{
    echo "Deletion Plan Summary"
    echo "===================="
    echo "Generated from audit: $LATEST_AUDIT"
    echo "Generated at: $(date)"
    echo ""
    echo "File counts:"
    echo "  - Likely trash candidates: $TRASH_COUNT"
    echo "  - Large non-source files (top 500): $LARGE_COUNT"
    echo "  - Duplicate file candidates: $DUPLICATE_COUNT"
    echo ""
    echo "Plan files:"
    echo "  - 01_likely_trash_candidates.txt"
    echo "  - 02_large_non_source_files.txt"
    echo "  - 03_duplicate_files.txt"
    echo ""
    echo "NOTE: This is a read-only analysis. No files have been deleted."
    echo "Review the plan files and use quarantine_move.sh to safely move files."
} | tee "$PLAN_DIR/summary.txt"

echo ""
echo "Deletion plan generated in: $PLAN_DIR"
