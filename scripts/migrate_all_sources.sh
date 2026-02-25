#!/bin/bash
# Comprehensive Source File Migration - ALL core library files
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_FILE="$PROJECT_ROOT/MIGRATION_BACKUP_LOG.md"
echo "" >> "$LOG_FILE"
echo "## Additional Core Library Files Migration" >> "$LOG_FILE"
echo "**Started:** $(date)" >> "$LOG_FILE"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

checksum() {
    if command -v shasum &> /dev/null; then
        shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v sha256sum &> /dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        echo "unknown"
    fi
}

safe_copy() {
    local src="$1"
    local dst="$2"
    local desc="$3"
    
    if [ ! -f "$src" ]; then
        log "⚠️  Source not found: $src (skipping)"
        return
    fi
    
    if [ -f "$dst" ]; then
        log "⚠️  Destination exists: $dst (skipping to avoid overwrite)"
        return
    fi
    
    local src_checksum=$(checksum "$src")
    log "Copying $desc: $(basename "$src")"
    
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    local dst_checksum=$(checksum "$dst")
    
    if [ "$src_checksum" != "$dst_checksum" ]; then
        log "  ❌ Checksum mismatch!"
        return 1
    fi
    log "  ✅ Verified"
}

log "Migrating ALL core library files..."

# Core directory files (excluding intent_ir which already exists)
find KmiDi_FINAL/engine/src/core -type f \( -name "*.cpp" -o -name "*.h" \) ! -path "*/intent_ir/*" 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "core"
done

# Audio directory
find KmiDi_FINAL/engine/src/audio -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "audio"
done

# Biometric directory (including .mm files)
find KmiDi_FINAL/engine/src/biometric -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.mm" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "biometric"
done

log "Migration complete!"
echo "**Completed:** $(date)" >> "$LOG_FILE"

# Music theory directory
find KmiDi_FINAL/engine/src/music_theory -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "music_theory"
done

# UI directory
find KmiDi_FINAL/engine/src/ui -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "ui"
done

# Kelly directory
find KmiDi_FINAL/engine/src/kelly -type f \( -name "*.cpp" -o -name "*.h" \) ! -path "*/__pycache__/*" 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "kelly"
done

# ML directory
find KmiDi_FINAL/engine/src/ml -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "ml"
done

# MIDI directory
find KmiDi_FINAL/engine/src/midi -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "midi"
done

# Harmony directory
find KmiDi_FINAL/engine/src/harmony -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "harmony"
done

# Groove directory
find KmiDi_FINAL/engine/src/groove -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "groove"
done

# Engines directory
find KmiDi_FINAL/engine/src/engines -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "engines"
done

# Diagnostics directory
find KmiDi_FINAL/engine/src/diagnostics -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "diagnostics"
done

# OSC directory
find KmiDi_FINAL/engine/src/osc -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "osc"
done

# Common directory
find KmiDi_FINAL/engine/src/common -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "common"
done

# Components directory
find KmiDi_FINAL/engine/src/components -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "components"
done

# Engine directory
find KmiDi_FINAL/engine/src/engine -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "engine"
done

# Export directory
find KmiDi_FINAL/engine/src/export -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "export"
done

# Hooks directory
find KmiDi_FINAL/engine/src/hooks -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "hooks"
done

# Learning directory
find KmiDi_FINAL/engine/src/learning -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "learning"
done

# Project directory
find KmiDi_FINAL/engine/src/project -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "project"
done

# PRROT directory
find KmiDi_FINAL/engine/src/prrot -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "prrot"
done

# Python directory
find KmiDi_FINAL/engine/src/python -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "python"
done

# Voice directory
find KmiDi_FINAL/engine/src/voice -type f \( -name "*.cpp" -o -name "*.h" \) 2>/dev/null | while read src; do
    rel_path=${src#KmiDi_FINAL/engine/src/}
    dst="src/$rel_path"
    safe_copy "$src" "$dst" "voice"
done

log "All directories migration complete!"
