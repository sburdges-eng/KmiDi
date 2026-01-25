#!/usr/bin/env bash
set -e

# Default to Documents folder, but allow override
DOCS_DIR="${1:-$HOME/Documents}"
OUT="_DOCUMENTS_AUDIT_$(date +%Y%m%d_%H%M%S)"

if [ ! -d "$DOCS_DIR" ]; then
  echo "Error: Directory '$DOCS_DIR' does not exist"
  exit 1
fi

mkdir -p "$OUT"

echo "Auditing documents at: $DOCS_DIR"
echo "Outputting to: $OUT"
echo

# Get total size
TOTAL_SIZE=$(du -sh "$DOCS_DIR" 2>/dev/null | cut -f1)
echo "Total size: $TOTAL_SIZE"
echo

############################
# 1. Top-level directory sizes
############################
echo "Analyzing top-level directories..."
du -h -d 1 "$DOCS_DIR" 2>/dev/null | \
  sort -hr | \
  head -50 > "$OUT/top_level_sizes.txt"

############################
# 2. File count per top-level directory
############################
echo "Counting files per top-level directory..."
find "$DOCS_DIR" -mindepth 1 -maxdepth 1 -type d -exec bash -c '
  for d; do
    count=$(find "$d" -type f 2>/dev/null | wc -l | tr -d " ")
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    printf "%12s  %8s  %s\n" "$count" "$size" "$d"
  done
' bash {} + | sort -rn > "$OUT/dirs_by_file_count.txt"

############################
# 3. Largest files (top 100)
############################
echo "Finding largest files..."
find "$DOCS_DIR" -type f -exec du -h {} + 2>/dev/null | \
  sort -hr | \
  head -100 > "$OUT/largest_files.txt"

############################
# 4. File extensions breakdown
############################
echo "Analyzing file types..."
find "$DOCS_DIR" -type f 2>/dev/null | \
  sed 's/.*\.//' | \
  sed 's/.*\///' | \
  grep -v '^$' | \
  sort | \
  uniq -c | \
  sort -nr | \
  head -50 > "$OUT/file_extensions.txt"

############################
# 5. Large directories (recursive, top 30)
############################
echo "Finding largest directories (recursive)..."
du -h "$DOCS_DIR" 2>/dev/null | \
  sort -hr | \
  head -30 > "$OUT/largest_directories.txt"

############################
# 6. Likely large media/video files
############################
echo "Finding large media files..."
{
  find "$DOCS_DIR" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.wmv" \) -exec du -h {} + 2>/dev/null | sort -hr | head -50
} > "$OUT/large_video_files.txt" || touch "$OUT/large_video_files.txt"

############################
# 7. Large image files
############################
echo "Finding large image files..."
{
  find "$DOCS_DIR" -type f \( -iname "*.psd" -o -iname "*.raw" -o -iname "*.cr2" -o -iname "*.nef" -o -iname "*.tiff" -o -iname "*.tif" \) -exec du -h {} + 2>/dev/null | sort -hr | head -50
} > "$OUT/large_image_files.txt" || touch "$OUT/large_image_files.txt"

############################
# 8. Archive files
############################
echo "Finding archive files..."
{
  find "$DOCS_DIR" -type f \( -iname "*.zip" -o -iname "*.tar" -o -iname "*.gz" -o -iname "*.7z" -o -iname "*.rar" -o -iname "*.dmg" \) -exec du -h {} + 2>/dev/null | sort -hr | head -50
} > "$OUT/archive_files.txt" || touch "$OUT/archive_files.txt"

############################
# 9. Potential duplicates (by name)
############################
echo "Checking for duplicate filenames..."
find "$DOCS_DIR" -type f 2>/dev/null | \
  xargs -I{} basename "{}" | \
  sort | \
  uniq -d | \
  head -100 > "$OUT/duplicate_filenames.txt" || touch "$OUT/duplicate_filenames.txt"

############################
# 10. Summary
############################
{
  echo "==== DOCUMENTS AUDIT SUMMARY ===="
  echo
  echo "Directory: $DOCS_DIR"
  echo "Total size: $TOTAL_SIZE"
  echo
  echo "Top 10 largest top-level directories:"
  head -10 "$OUT/top_level_sizes.txt"
  echo
  echo "Top 10 directories by file count:"
  head -10 "$OUT/dirs_by_file_count.txt"
  echo
  echo "Top 10 file types:"
  head -10 "$OUT/file_extensions.txt"
  echo
  echo "Top 10 largest files:"
  head -10 "$OUT/largest_files.txt"
  echo
  echo "Top 10 largest directories (recursive):"
  head -10 "$OUT/largest_directories.txt"
  echo
  echo "Large video files found:"
  wc -l < "$OUT/large_video_files.txt"
  echo
  echo "Large image files found:"
  wc -l < "$OUT/large_image_files.txt"
  echo
  echo "Archive files found:"
  wc -l < "$OUT/archive_files.txt"
  echo
  echo "Potential duplicate filenames:"
  wc -l < "$OUT/duplicate_filenames.txt"
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
echo
echo "Detailed reports saved to: $OUT/"
echo "Read $OUT/summary.txt for overview"
