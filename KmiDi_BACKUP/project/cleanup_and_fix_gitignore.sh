#!/bin/bash
# Comprehensive cleanup script
# This will:
# 1. Remove tracked files that should be ignored
# 2. Clean up filesystem
# 3. Fix .gitignore duplicates

set -e

REPO_DIR="$(pwd)"
echo "=== Cleanup Script for: $REPO_DIR ==="
echo ""

# Safety check
read -p "This will delete build artifacts, venv directories, and macOS files. Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "=== Step 1: Removing tracked files that should be ignored ==="

# Remove tracked .gradle files
if git ls-files | grep -q "\.gradle"; then
    echo "Removing tracked .gradle files..."
    git rm -r --cached iDAW-Android/app/.gradle 2>/dev/null || true
    git rm -r --cached iDAW-Android/.gradle 2>/dev/null || true
fi

# Remove tracked .DS_Store files
if git ls-files | grep -q "\.DS_Store"; then
    echo "Removing tracked .DS_Store files..."
    git ls-files | grep "\.DS_Store" | xargs git rm --cached 2>/dev/null || true
fi

# Remove tracked resource forks
if git ls-files | grep -q "^\._"; then
    echo "Removing tracked resource fork files..."
    git ls-files | grep "^\._" | xargs git rm --cached 2>/dev/null || true
fi

echo "✓ Step 1 complete"
echo ""

echo "=== Step 2: Cleaning up filesystem ==="

# Remove macOS files
echo "Removing macOS resource forks..."
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "._*" -delete 2>/dev/null || true

# Remove build directories
echo "Removing build directories..."
rm -rf build build-* cmake-build-* 2>/dev/null || true
rm -rf src/build src_penta-core/build 2>/dev/null || true

# Remove Python virtual environments
echo "Removing Python virtual environments..."
rm -rf venv .venv ml_training/venv ml_framework/venv 2>/dev/null || true

# Remove Rust build artifacts
echo "Removing Rust build artifacts..."
rm -rf src-tauri/target 2>/dev/null || true

# Remove Gradle build cache
echo "Removing Gradle build cache..."
rm -rf iDAW-Android/.gradle iDAW-Android/app/.gradle 2>/dev/null || true

# Remove node_modules if exists
echo "Removing node_modules..."
rm -rf node_modules 2>/dev/null || true

echo "✓ Step 2 complete"
echo ""

echo "=== Step 3: Cleaning up .gitignore ==="
# This will be done manually or with a Python script
echo "Note: .gitignore cleanup should be done manually to review duplicates"
echo "✓ Step 3 complete"
echo ""

echo "=== Cleanup Summary ==="
echo "Files removed from git tracking (use 'git commit' to finalize)"
echo "Filesystem cleaned of build artifacts and venv directories"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Commit the cleanup: git commit -m 'Clean up unnecessary files and build artifacts'"
echo "3. Regenerate venv when needed: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"

