#!/bin/bash
set -e

# Project Audit Script
# Analyzes project structure, file counts, disk usage, and detects duplicates

if [ $# -lt 1 ]; then
    echo "Usage: $0 <project_root>"
    exit 1
fi

PROJECT_ROOT="$1"
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: $PROJECT_ROOT is not a directory"
    exit 1
fi

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="_AUDIT_REPORT_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Starting audit of $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# 1. Total file count
echo "Calculating total file count..."
TOTAL_FILES=$(find . -type f | wc -l | tr -d ' ')
echo "Total files: $TOTAL_FILES" | tee "$OUTPUT_DIR/summary.txt"
echo "" | tee -a "$OUTPUT_DIR/summary.txt"

# 2. File counts per top-level folder
echo "Analyzing files per top-level folder..."
{
    echo "Files per top-level folder:"
    echo "============================"
    for dir in */; do
        if [ -d "$dir" ]; then
            count=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
            printf "%-40s %10s\n" "$dir" "$count"
        fi
    done
} | tee "$OUTPUT_DIR/files_per_folder.txt"

# 3. Disk usage per folder
echo "Calculating disk usage per folder..."
{
    echo "Disk usage per top-level folder:"
    echo "================================="
    du -sh */ 2>/dev/null | sort -hr | while read size dir; do
        printf "%-40s %10s\n" "$dir" "$size"
    done
} | tee "$OUTPUT_DIR/disk_usage_per_folder.txt"

# 4. File extension breakdown
echo "Analyzing file extensions..."
{
    echo "File extension frequency:"
    echo "========================"
    find . -type f -name '*.*' | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -100 | while read count ext; do
        printf "%-20s %10s\n" ".$ext" "$count"
    done
} | tee "$OUTPUT_DIR/file_extensions.txt"

# 5. Likely trash files
echo "Detecting likely trash files..."
{
    echo "Likely trash files (cache, build artifacts, etc.):"
    echo "==================================================="
    find . -type f \( \
        -name "*.pyc" -o \
        -name "*.pyo" -o \
        -name "__pycache__" -o \
        -name "*.class" -o \
        -name "*.o" -o \
        -name "*.obj" -o \
        -name "*.exe" -o \
        -name "*.dll" -o \
        -name "*.so" -o \
        -name "*.dylib" -o \
        -name "*.a" -o \
        -name "*.lib" -o \
        -name "CMakeCache.txt" -o \
        -name "cmake_install.cmake" -o \
        -name "Makefile" -o \
        -name "*.log" -o \
        -name "*.tmp" -o \
        -name "*.temp" -o \
        -name "*.swp" -o \
        -name "*.swo" -o \
        -name "*~" -o \
        -name ".DS_Store" -o \
        -name "Thumbs.db" -o \
        -name ".idea/" -o \
        -name ".vscode/" -o \
        -name "node_modules/" -o \
        -path "*/build/*" -o \
        -path "*/_build/*" -o \
        -path "*/dist/*" -o \
        -path "*/.pytest_cache/*" -o \
        -path "*/__pycache__/*" \
    \) 2>/dev/null | sort
} | tee "$OUTPUT_DIR/likely_trash_files.txt"

# 6. Duplicate file detection (hash-based)
echo "Detecting duplicate files (this may take a while)..."
{
    echo "Duplicate files (grouped by hash):"
    echo "==================================="
    # Create temporary file with all hashes
    TEMP_HASH_FILE=$(mktemp)
    find . -type f -exec shasum -a 256 {} \; 2>/dev/null | sort > "$TEMP_HASH_FILE"
    
    # Find duplicate hashes using awk (macOS compatible)
    awk '{print $1}' "$TEMP_HASH_FILE" | sort | uniq -d | while read hash; do
        echo "=== Hash: $hash ==="
        grep "^$hash" "$TEMP_HASH_FILE" | cut -d' ' -f2- | while read file; do
            echo "  $file"
        done
        echo ""
    done
    
    rm -f "$TEMP_HASH_FILE"
} | tee "$OUTPUT_DIR/duplicate_hashes.txt"

# Update summary
{
    echo "Audit Summary"
    echo "============="
    echo "Total files: $TOTAL_FILES"
    echo "Timestamp: $TIMESTAMP"
    echo ""
    echo "Report files generated:"
    echo "  - summary.txt"
    echo "  - files_per_folder.txt"
    echo "  - disk_usage_per_folder.txt"
    echo "  - file_extensions.txt"
    echo "  - likely_trash_files.txt"
    echo "  - duplicate_hashes.txt"
} > "$OUTPUT_DIR/summary.txt"

echo ""
echo "Audit complete! Results saved to: $OUTPUT_DIR"
