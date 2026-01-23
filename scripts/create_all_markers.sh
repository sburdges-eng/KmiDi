#!/bin/bash
# Create ACTIVE_DEVELOPMENT.md markers for all directories
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

create_marker() {
    local dir="$1"
    local name="$2"
    local desc="$3"
    
    if [ ! -d "$dir" ]; then
        return
    fi
    
    if [ -f "$dir/ACTIVE_DEVELOPMENT.md" ]; then
        return
    fi
    
    cat > "$dir/ACTIVE_DEVELOPMENT.md" <<EOF
# 🟡 ACTIVE DEVELOPMENT

**Status:** 🟡 ACTIVE
**Directory:** \`$dir\`
**Color Code:** Yellow/Gold (🟡)

## Purpose

$desc

## Migration

Migrated from \`KmiDi_FINAL/engine/src/$name/\` on 2026-01-21.
EOF
    echo "Created marker: $dir/ACTIVE_DEVELOPMENT.md"
}

create_marker "src/audio" "audio" "Audio analysis and processing components"
create_marker "src/biometric" "biometric" "Biometric input processing (including Objective-C++ files)"
create_marker "src/music_theory" "music_theory" "Music theory engines (harmony, rhythm, knowledge graph)"
create_marker "src/ui" "ui" "UI components and workstation panels"
create_marker "src/ml" "ml" "Machine learning components"
create_marker "src/midi" "midi" "MIDI processing and sequencing"
create_marker "src/harmony" "harmony" "Harmony analysis and voice leading"
create_marker "src/groove" "groove" "Groove and rhythm processing"
create_marker "src/diagnostics" "diagnostics" "Performance diagnostics and monitoring"
create_marker "src/osc" "osc" "OSC (Open Sound Control) communication"
create_marker "src/common" "common" "Common utilities and shared code"
create_marker "src/components" "components" "Reusable components"
create_marker "src/engine" "engine" "Engine core functionality"
create_marker "src/engines" "engines" "Engine implementations"
create_marker "src/export" "export" "Export functionality"
create_marker "src/learning" "learning" "Learning systems"
create_marker "src/project" "project" "Project management"
create_marker "src/prrot" "prrot" "PRROT system"
create_marker "src/python" "python" "Python bindings and integration"
create_marker "src/voice" "voice" "Voice processing and synthesis"
[ -d "src/kelly" ] && create_marker "src/kelly" "kelly" "Kelly-specific implementations"

echo "All markers created!"
