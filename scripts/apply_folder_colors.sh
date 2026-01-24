#!/bin/bash
# Apply Yellow/Gold Finder Labels to Active Development Folders
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

YELLOW_LABEL=3  # macOS Finder label: 3=Yellow

log() { echo "[$(date '+%H:%M:%S')] $1"; }

apply_label() {
    local path="$1"
    local name="$2"
    
    if [ ! -e "$path" ]; then
        log "⚠️  Path not found: $path (skipping)"
        return
    fi
    
    local abs_path="$PROJECT_ROOT/$path"
    
    log "Applying yellow/gold label to: $name"
    
    osascript <<EOF 2>/dev/null || true
tell application "Finder"
    set theItem to POSIX file "$abs_path" as alias
    set label index of theItem to $YELLOW_LABEL
end tell
EOF
    
    [ $? -eq 0 ] && log "  ✅ Label applied" || log "  ⚠️  May need Finder permissions"
}

log "Applying yellow/gold Finder labels..."

apply_label "src" "src/ (Main source)"
apply_label "src/plugin" "src/plugin/ (Plugin)"
apply_label "src/gui" "src/gui/ (GUI)"
apply_label "src/bridge" "src/bridge/ (Bridge)"
apply_label "src/core" "src/core/ (Core)"
[ -d "include" ] && apply_label "include" "include/ (Headers)"
[ -d "tests" ] && apply_label "tests" "tests/ (Tests)"

apply_label "PROJECT_SOURCE_MANIFEST.md" "Manifest"
apply_label "src/INDEX.md" "Index"
apply_label "PROJECT_DIRECTORY_MAP.md" "Directory Map"
apply_label "MIGRATION_COMPLETE_SUMMARY.md" "Summary"

log "Complete! Check Finder to see yellow/gold labels."

# Apply labels to all new directories
for dir in audio biometric music_theory ui ml midi harmony groove diagnostics osc common components engine engines export learning project prrot python voice kelly; do
    [ -d "src/$dir" ] && apply_label "src/$dir" "src/$dir/ ($dir)"
done

log "All directory labels applied!"
