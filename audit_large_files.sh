#!/usr/bin/env bash
set -e

# Audit large files across common locations
START_DIR="${1:-$HOME}"
OUT="_LARGE_FILES_AUDIT_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT"

echo "Auditing large files starting from: $START_DIR"
echo "Outputting to: $OUT"
echo

############################
# 1. Check common directories
############################
echo "Checking common directories..."
{
  echo "=== COMMON DIRECTORIES ==="
  for dir in Documents Downloads Desktop Movies Music Pictures Library; do
    if [ -d "$HOME/$dir" ]; then
      size=$(du -sh "$HOME/$dir" 2>/dev/null | cut -f1)
      files=$(find "$HOME/$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
      printf "%-20s %10s  %8s files\n" "$dir" "$size" "$files"
    fi
  done
} > "$OUT/common_dirs.txt"

cat "$OUT/common_dirs.txt"
echo

############################
# 2. Largest directories in home (top 20)
############################
echo "Finding largest directories in $START_DIR..."
du -h -d 1 "$START_DIR" 2>/dev/null | \
  sort -hr | \
  head -20 > "$OUT/largest_home_dirs.txt"

echo "Top 10 largest directories:"
head -10 "$OUT/largest_home_dirs.txt"
echo

############################
# 3. Find files larger than 1GB
############################
echo "Finding files larger than 1GB..."
find "$START_DIR" -type f -size +1G -exec du -h {} + 2>/dev/null | \
  sort -hr > "$OUT/files_over_1gb.txt" || touch "$OUT/files_over_1gb.txt"

if [ -s "$OUT/files_over_1gb.txt" ]; then
  echo "Files over 1GB found:"
  wc -l < "$OUT/files_over_1gb.txt"
  head -20 "$OUT/files_over_1gb.txt"
else
  echo "No files over 1GB found"
fi
echo

############################
# 4. Find files larger than 100MB
############################
echo "Finding files larger than 100MB..."
find "$START_DIR" -type f -size +100M -exec du -h {} + 2>/dev/null | \
  sort -hr | \
  head -100 > "$OUT/files_over_100mb.txt" || touch "$OUT/files_over_100mb.txt"

echo "Files over 100MB (showing top 20):"
head -20 "$OUT/files_over_100mb.txt"
echo

############################
# 5. Check Library caches and logs
############################
echo "Checking Library caches..."
{
  if [ -d "$HOME/Library" ]; then
    echo "=== LIBRARY SUBDIRECTORIES ==="
    du -h -d 1 "$HOME/Library" 2>/dev/null | \
      sort -hr | \
      head -20
  fi
} > "$OUT/library_dirs.txt" || touch "$OUT/library_dirs.txt"

############################
# 6. Check for large virtual machines, containers
############################
echo "Checking for VMs, containers, node_modules..."
{
  find "$START_DIR" -type d \( -iname "*.app" -o -iname "node_modules" -o -iname ".venv" -o -iname "venv" -o -iname "*.vmdk" -o -iname "*.vdi" \) 2>/dev/null | \
    head -50 | \
    xargs -I{} du -sh "{}" 2>/dev/null | \
    sort -hr
} > "$OUT/potential_large_dirs.txt" || touch "$OUT/potential_large_dirs.txt"

############################
# 7. Summary
############################
{
  echo "==== LARGE FILES AUDIT SUMMARY ===="
  echo
  echo "Audited from: $START_DIR"
  echo "Date: $(date)"
  echo
  echo "Common directories:"
  cat "$OUT/common_dirs.txt"
  echo
  echo "Top 10 largest directories:"
  head -10 "$OUT/largest_home_dirs.txt"
  echo
  echo "Files over 1GB:"
  if [ -s "$OUT/files_over_1gb.txt" ]; then
    wc -l < "$OUT/files_over_1gb.txt"
    echo "See: $OUT/files_over_1gb.txt"
  else
    echo "0 found"
  fi
  echo
  echo "Files over 100MB (top 10):"
  head -10 "$OUT/files_over_100mb.txt"
  echo
  echo "Potential large directories (VMs, node_modules, etc.):"
  head -10 "$OUT/potential_large_dirs.txt"
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
echo
echo "Detailed reports saved to: $OUT/"
