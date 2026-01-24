#!/usr/bin/env bash
set -e

LIBRARY_DIR="$HOME/Library"
OUT="_LIBRARY_AUDIT_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT"

echo "Analyzing Library directory: $LIBRARY_DIR"
echo "Outputting to: $OUT"
echo

############################
# 1. Top-level Library subdirectories
############################
echo "Analyzing Library subdirectories..."
du -h -d 1 "$LIBRARY_DIR" 2>/dev/null | \
  sort -hr > "$OUT/library_subdirs.txt"

echo "Top 20 largest Library subdirectories:"
head -20 "$OUT/library_subdirs.txt"
echo

############################
# 2. Common large Library locations
############################
echo "Checking common large locations..."

# Caches
if [ -d "$LIBRARY_DIR/Caches" ]; then
  echo "=== CACHES ==="
  du -h -d 1 "$LIBRARY_DIR/Caches" 2>/dev/null | sort -hr | head -20
  echo
fi

# Application Support
if [ -d "$LIBRARY_DIR/Application Support" ]; then
  echo "=== APPLICATION SUPPORT ==="
  du -h -d 1 "$LIBRARY_DIR/Application Support" 2>/dev/null | sort -hr | head -20
  echo
fi

# Containers (for sandboxed apps)
if [ -d "$LIBRARY_DIR/Containers" ]; then
  echo "=== CONTAINERS ==="
  du -h -d 1 "$LIBRARY_DIR/Containers" 2>/dev/null | sort -hr | head -20
  echo
fi

# Group Containers
if [ -d "$LIBRARY_DIR/Group Containers" ]; then
  echo "=== GROUP CONTAINERS ==="
  du -h -d 1 "$LIBRARY_DIR/Group Containers" 2>/dev/null | sort -hr | head -20
  echo
fi

# Logs
if [ -d "$LIBRARY_DIR/Logs" ]; then
  echo "=== LOGS ==="
  du -h -d 1 "$LIBRARY_DIR/Logs" 2>/dev/null | sort -hr | head -20
  echo
fi

# Developer (Xcode, etc.)
if [ -d "$LIBRARY_DIR/Developer" ]; then
  echo "=== DEVELOPER ==="
  du -h -d 1 "$LIBRARY_DIR/Developer" 2>/dev/null | sort -hr | head -20
  echo
fi

############################
# 3. Summary
############################
{
  echo "==== LIBRARY AUDIT SUMMARY ===="
  echo
  echo "Total Library size: $(du -sh "$LIBRARY_DIR" 2>/dev/null | cut -f1)"
  echo
  echo "Top 20 largest subdirectories:"
  head -20 "$OUT/library_subdirs.txt"
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
echo
echo "Full report: $OUT/library_subdirs.txt"
