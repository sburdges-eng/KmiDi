#!/usr/bin/env bash
set -e

OUT="_PROJECT_DEPENDENCY_CHECK_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

echo "Checking if any projects/workspaces need the large directories..."
echo "Outputting to: $OUT"
echo

# Large directories we identified
LARGE_DIRS=(
  "$HOME/Library/Application Support/CloudDocs"
  "$HOME/Library/Application Support/MobileSync"
  "$HOME/Library/Group Containers/UBF8T346G9.OneDriveStandaloneSuite"
  "$HOME/Library/Application Support/com.apple.wallpaper"
  "$HOME/Library/Caches"
  "$HOME/RECOVERY_OPS"
  "$HOME/audio"
  "$HOME/.cursor"
  "$HOME/Desktop"
)

############################
# 1. Find all project directories
############################
echo "Finding project directories..."
PROJECT_DIRS="$OUT/project_directories.txt"
{
  # Common project locations
  find "$HOME" -maxdepth 3 -type d \( \
    -name ".git" -o \
    -name "node_modules" -o \
    -name ".venv" -o \
    -name "venv" -o \
    -name "build" -o \
    -name "target" \
  \) -exec dirname {} \; 2>/dev/null | \
    sort -u | \
    head -100
} > "$PROJECT_DIRS"

PROJECT_COUNT=$(wc -l < "$PROJECT_DIRS" | tr -d ' ')
echo "Found $PROJECT_COUNT potential project directories"
echo

############################
# 2. Check for symlinks to large directories
############################
echo "Checking for symlinks to large directories..."
SYMLINKS_FOUND="$OUT/symlinks_to_large_dirs.txt"
> "$SYMLINKS_FOUND"

for large_dir in "${LARGE_DIRS[@]}"; do
  if [ -d "$large_dir" ]; then
    # Find symlinks pointing to this directory
    find "$HOME" -type l -exec readlink -f {} \; 2>/dev/null | \
      grep -q "$large_dir" && \
      echo "Symlinks found pointing to: $large_dir" >> "$SYMLINKS_FOUND" || true
    
    # Also check if directory itself is a symlink
    if [ -L "$large_dir" ]; then
      echo "IS SYMLINK: $large_dir -> $(readlink "$large_dir")" >> "$SYMLINKS_FOUND"
    fi
  fi
done

if [ -s "$SYMLINKS_FOUND" ]; then
  echo "⚠️  Symlinks found - review needed:"
  cat "$SYMLINKS_FOUND"
else
  echo "✅ No symlinks found pointing to large directories"
fi
echo

############################
# 3. Search project files for references to Library paths
############################
echo "Searching project files for Library path references..."
PROJECT_REFS="$OUT/project_references.txt"
> "$PROJECT_REFS"

# Common config files that might reference paths
CONFIG_PATTERNS=("*.json" "*.yaml" "*.yml" "*.toml" "*.config" "*.conf" "*.env" "*.sh" "*.py" "*.js" "*.ts" "Makefile" "CMakeLists.txt" ".gitignore")

for large_dir in "${LARGE_DIRS[@]}"; do
  if [ -d "$large_dir" ]; then
    dir_name=$(basename "$large_dir")
    # Search in project directories
    while IFS= read -r proj_dir; do
      for pattern in "${CONFIG_PATTERNS[@]}"; do
        find "$proj_dir" -maxdepth 3 -type f -name "$pattern" 2>/dev/null | \
          xargs grep -l "$dir_name\|$large_dir" 2>/dev/null | \
          head -5 >> "$PROJECT_REFS" || true
      done
    done < "$PROJECT_DIRS"
  fi
done

if [ -s "$PROJECT_REFS" ]; then
  echo "⚠️  Project files referencing large directories:"
  sort -u "$PROJECT_REFS" | head -20
  echo "  (See $PROJECT_REFS for full list)"
else
  echo "✅ No project files found referencing large directories"
fi
echo

############################
# 4. Check specific known project locations
############################
echo "Checking known project locations..."
KNOWN_PROJECTS="$OUT/known_projects.txt"
> "$KNOWN_PROJECTS"

KNOWN_LOCATIONS=(
  "$HOME/Documents"
  "$HOME/Desktop"
  "$HOME/projects"
  "$HOME/code"
  "$HOME/workspace"
  "$HOME/KmiDi"
  "$HOME/KmiDi-1"
  "$HOME/RECOVERY_OPS"
  "$HOME/audio"
)

for loc in "${KNOWN_LOCATIONS[@]}"; do
  if [ -d "$loc" ]; then
    # Check if it's a git repo or has project files
    if [ -d "$loc/.git" ] || [ -f "$loc/package.json" ] || [ -f "$loc/CMakeLists.txt" ] || [ -f "$loc/Cargo.toml" ] || [ -f "$loc/requirements.txt" ]; then
      echo "$loc" >> "$KNOWN_PROJECTS"
      
      # Check if this project references Library paths
      for large_dir in "${LARGE_DIRS[@]}"; do
        if find "$loc" -type f \( -name "*.py" -o -name "*.js" -o -name "*.json" -o -name "*.yaml" \) -exec grep -l "$large_dir\|Library/Application Support\|Library/Group Containers" {} \; 2>/dev/null | head -1 | grep -q .; then
          echo "  ⚠️  References: $large_dir" >> "$KNOWN_PROJECTS"
        fi
      done
    fi
  fi
done

if [ -s "$KNOWN_PROJECTS" ]; then
  echo "Known projects found:"
  cat "$KNOWN_PROJECTS"
fi
echo

############################
# 5. Check for hardcoded paths in common configs
############################
echo "Checking common config files for hardcoded paths..."
CONFIG_REFS="$OUT/config_references.txt"
> "$CONFIG_REFS"

# Check common config locations
CONFIG_LOCATIONS=(
  "$HOME/.zshrc"
  "$HOME/.bashrc"
  "$HOME/.bash_profile"
  "$HOME/.config"
  "$HOME/.vscode"
  "$HOME/.cursor"
)

for config_loc in "${CONFIG_LOCATIONS[@]}"; do
  if [ -f "$config_loc" ] || [ -d "$config_loc" ]; then
    grep -r "Library/Application Support\|Library/Group Containers\|CloudDocs\|MobileSync\|OneDriveStandaloneSuite" "$config_loc" 2>/dev/null | \
      head -10 >> "$CONFIG_REFS" || true
  fi
done

if [ -s "$CONFIG_REFS" ]; then
  echo "⚠️  Config files referencing Library paths:"
  cat "$CONFIG_REFS" | head -10
else
  echo "✅ No config files found referencing Library paths"
fi
echo

############################
# 6. Summary
############################
{
  echo "==== PROJECT DEPENDENCY CHECK SUMMARY ===="
  echo
  echo "Projects found: $PROJECT_COUNT"
  echo
  echo "Symlinks to large directories:"
  if [ -s "$SYMLINKS_FOUND" ]; then
    wc -l < "$SYMLINKS_FOUND"
    echo "Review: $SYMLINKS_FOUND"
  else
    echo "None found"
  fi
  echo
  echo "Project files referencing large directories:"
  if [ -s "$PROJECT_REFS" ]; then
    unique_refs=$(sort -u "$PROJECT_REFS" | wc -l | tr -d ' ')
    echo "$unique_refs unique files"
    echo "Review: $PROJECT_REFS"
  else
    echo "None found"
  fi
  echo
  echo "Config files referencing Library paths:"
  if [ -s "$CONFIG_REFS" ]; then
    wc -l < "$CONFIG_REFS"
    echo "Review: $CONFIG_REFS"
  else
    echo "None found"
  fi
  echo
  echo "=== SAFETY ASSESSMENT ==="
  echo
  if [ ! -s "$SYMLINKS_FOUND" ] && [ ! -s "$PROJECT_REFS" ] && [ ! -s "$CONFIG_REFS" ]; then
    echo "✅ SAFE: No projects appear to depend on large Library directories"
    echo "   Caches and backups can likely be safely cleaned"
    echo "   iCloud/OneDrive sync - review manually for your workflow needs"
  else
    echo "⚠️  CAUTION: Some references found"
    echo "   Review the files listed above before cleaning"
  fi
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
echo
echo "Detailed reports in: $OUT/"
