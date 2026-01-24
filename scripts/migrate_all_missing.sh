#!/bin/bash
# Migrate ALL missing files: headers, cpp_music_brain, etc.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_FILE="$PROJECT_ROOT/MIGRATION_BACKUP_LOG.md"
echo "" >> "$LOG_FILE"
echo "## Complete Missing Files Migration" >> "$LOG_FILE"
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
    return
  fi

  if [ -f "$dst" ]; then
    log "⚠️  EXISTS: $dst (skipping)"
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

log "=== Migrating include/ headers ==="

# Migrate engine/include to include/
find KmiDi_FINAL/engine/include -type f \( -name "*.h" -o -name "*.hpp" \) 2> /dev/null | while read src; do
  rel_path=${src#KmiDi_FINAL/engine/include/}
  dst="include/$rel_path"
  safe_copy "$src" "$dst" "header"
done

# Migrate shared/include to include/ (merge)
find KmiDi_FINAL/shared/include -type f \( -name "*.h" -o -name "*.hpp" \) 2> /dev/null | while read src; do
  rel_path=${src#KmiDi_FINAL/shared/include/}
  dst="include/$rel_path"
  safe_copy "$src" "$dst" "shared-header"
done

log "=== Migrating cpp_music_brain source files ==="

# Migrate cpp_music_brain/src to src/cpp_music_brain/
find KmiDi_FINAL/engine/cpp_music_brain/src -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) 2> /dev/null | while read src; do
  rel_path=${src#KmiDi_FINAL/engine/cpp_music_brain/src/}
  dst="src/cpp_music_brain/$rel_path"
  safe_copy "$src" "$dst" "cpp_music_brain"
done

# Migrate cpp_music_brain/include to include/
find KmiDi_FINAL/engine/cpp_music_brain/include -type f \( -name "*.h" -o -name "*.hpp" \) 2> /dev/null | while read src; do
  rel_path=${src#KmiDi_FINAL/engine/cpp_music_brain/include/}
  dst="include/$rel_path"
  safe_copy "$src" "$dst" "cpp_music_brain-header"
done

log "=== Migration complete! ==="
echo "**Completed:** $(date)" >> "$LOG_FILE"
