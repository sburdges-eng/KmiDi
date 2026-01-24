#!/bin/bash
# Migrate ONLY missing files - do not overwrite existing
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_FILE="$PROJECT_ROOT/MIGRATION_BACKUP_LOG.md"
echo "" >> "$LOG_FILE"
echo "## Missing Files Migration" >> "$LOG_FILE"
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
    log "⚠️  EXISTS: $dst (skipping to avoid overwrite)"
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

log "Migrating missing files only..."

# Bridge files
find KmiDi_FINAL/engine/src/bridge -type f \( -name "*.cpp" -o -name "*.h" \) 2> /dev/null | while read src; do
  rel_path=${src#KmiDi_FINAL/engine/src/}
  dst="src/$rel_path"
  safe_copy "$src" "$dst" "bridge"
done

# Plugin files
find KmiDi_FINAL/engine/src/plugin -type f \( -name "*.cpp" -o -name "*.h" \) 2> /dev/null | while read src; do
  rel_path=${src#KmiDi_FINAL/engine/src/}
  dst="src/$rel_path"
  safe_copy "$src" "$dst" "plugin"
done

# KellyML files
find KmiDi_FINAL/engine/src/KellyML -type f \( -name "*.cpp" -o -name "*.h" \) 2> /dev/null | while read src; do
  rel_path=${src#KmiDi_FINAL/engine/src/}
  dst="src/$rel_path"
  safe_copy "$src" "$dst" "KellyML"
done

# Any other missing files
find KmiDi_FINAL/engine/src -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.mm" \) ! -path "*/plugin/*" ! -path "*/gui/*" ! -path "*/bridge/*" ! -path "*/KellyML/*" ! -path "*/assets/*" ! -path "*/dsp/*" ! -name "App.*" ! -name "GameModel.*" ! -name "BridgeClient.*" ! -name "VoiceProcessor.*" ! -name "WavetableSynth.*" 2> /dev/null | while read src; do
  rel_path=${src#KmiDi_FINAL/engine/src/}
  dst="src/$rel_path"
  if [ ! -f "$dst" ]; then
    safe_copy "$src" "$dst" "core"
  fi
done

log "Missing files migration complete!"
echo "**Completed:** $(date)" >> "$LOG_FILE"
