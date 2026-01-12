#!/bin/bash
# Cleanup script for unnecessary files
# Run with: bash cleanup_unnecessary_files.sh

set -e

echo "=== Cleaning up unnecessary files ==="
echo ""

# 1. Remove macOS resource forks
echo "1. Removing macOS resource forks (.DS_Store, ._*)..."
find . -name ".DS_Store" -delete
find . -name "._*" -delete
echo "   ✓ Done"

# 2. Remove build directories (if they exist and are not needed)
echo ""
echo "2. Removing build directories..."
for dir in build build-* cmake-build-*; do
    if [ -d "$dir" ]; then
        echo "   Removing $dir"
        rm -rf "$dir"
    fi
done
echo "   ✓ Done"

# 3. Remove Python virtual environments
echo ""
echo "3. Removing Python virtual environments..."
for venv_dir in venv .venv ml_training/venv ml_framework/venv; do
    if [ -d "$venv_dir" ]; then
        echo "   Removing $venv_dir"
        rm -rf "$venv_dir"
    fi
done
echo "   ✓ 
# 4. Remove Rust build artifacts
echo ""
echo "4. Removing Rust build artifacts..."
if [ -d "src-tauri/target" ]; then
    echo "   Removing src-tauri/target"
    rm -rf src-tauri/target
fi
echo "   ✓ Done"

# 5. Remove Gradle build cache
echo ""
echo "5. Removing Gradle build cache..."
if [ -d "iDAW-Android/.gradle" ]; then
    echo "   Removing iDAW-Android/.gradle"
    rm -rf iDAW-Android/.gradle
fi
echo "   ✓ Done"

echo ""
echo "=== Cleanup complete ==="
echo "Note: These files are already in .gitignore and won't be committed"
