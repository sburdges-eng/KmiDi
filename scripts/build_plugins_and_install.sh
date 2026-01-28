#!/usr/bin/env bash
# =============================================================================
# Build Kelly VST3/CLAP plugins and install to user plug-in folders (macOS)
# =============================================================================
# Installs to:
#   ~/Library/Audio/Plug-Ins/VST3/
#   ~/Library/Audio/Plug-Ins/CLAP/
#
# Usage:
#   ./scripts/build_plugins_and_install.sh [Release|Debug]
#
# Requires: cmake, Qt6 (brew --prefix qt@6), build already configured.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_TYPE="${1:-Release}"

VST3_DIR="$HOME/Library/Audio/Plug-Ins/VST3"
CLAP_DIR="$HOME/Library/Audio/Plug-Ins/CLAP"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Kelly Plugin — Build & Install (VST3 / CLAP)                 ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$ROOT"

# Ensure CMake build exists and is configured for plugins
if [ ! -f "build/CMakeCache.txt" ]; then
  echo -e "${YELLOW}Configuring CMake (BUILD_PLUGINS=ON)...${NC}"
  QT_PREFIX=""
  command -v brew &>/dev/null && QT_PREFIX="$(brew --prefix qt@6 2>/dev/null)" || true
  [ -n "$QT_PREFIX" ] && export CMAKE_PREFIX_PATH="$QT_PREFIX"
  cmake -B build -S . -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DBUILD_PLUGINS=ON -DBUILD_DESKTOP=OFF
fi

echo -e "${YELLOW}Building plugins ($BUILD_TYPE)...${NC}"
cmake --build build --config "$BUILD_TYPE" --target KellyPlugin_VST3 KellyPlugin_CLAP 2>/dev/null \
  || cmake --build build --config "$BUILD_TYPE" -j8

ARTEFACTS="build/KellyPlugin_artefacts/$BUILD_TYPE"
if [ ! -d "$ARTEFACTS" ]; then
  # Fallback: any *artefacts* under build
  ARTEFACTS=$(find build -maxdepth 2 -type d -name '*artefacts*' 2>/dev/null | head -1)
fi

if [ -z "$ARTEFACTS" ] || [ ! -d "$ARTEFACTS" ]; then
  echo -e "${RED}Plugin artefacts not found under build/. Build may have failed.${NC}"
  exit 1
fi

mkdir -p "$VST3_DIR" "$CLAP_DIR"

install_one() {
  local src="$1"
  local dest_dir="$2"
  local name="$3"
  [ ! -e "$src" ] && return 0
  rm -rf "$dest_dir/$name"
  cp -R "$src" "$dest_dir/"
  xattr -cr "$dest_dir/$name" 2>/dev/null || true
  codesign --remove-signature "$dest_dir/$name" 2>/dev/null || true
  codesign --force --deep --sign - "$dest_dir/$name" 2>/dev/null || true
  echo -e "  ${GREEN}✓ Installed $name → $dest_dir${NC}"
}

# VST3
VST3_SRC=$(find "$ARTEFACTS" -type d -name "*.vst3" 2>/dev/null | head -1)
if [ -n "$VST3_SRC" ]; then
  echo -e "${YELLOW}Installing VST3...${NC}"
  install_one "$VST3_SRC" "$VST3_DIR" "$(basename "$VST3_SRC")"
fi

# CLAP (can be .clap bundle dir or .bundle on macOS)
CLAP_SRC=$(find "$ARTEFACTS" -type d \( -name "*.clap" -o -name "*.clap.bundle" \) 2>/dev/null | head -1)
if [ -z "$CLAP_SRC" ]; then
  CLAP_SRC=$(find "$ARTEFACTS" -type d -path "*CLAP*" -name "*.clap" 2>/dev/null | head -1)
fi
if [ -n "$CLAP_SRC" ]; then
  echo -e "${YELLOW}Installing CLAP...${NC}"
  install_one "$CLAP_SRC" "$CLAP_DIR" "$(basename "$CLAP_SRC")"
fi

if [ -z "$VST3_SRC" ] && [ -z "$CLAP_SRC" ]; then
  echo -e "${YELLOW}No VST3/CLAP bundles found in $ARTEFACTS. Paths may differ (e.g. product name).${NC}"
  echo "  Searched: $ARTEFACTS"
  exit 1
fi

echo ""
echo -e "${GREEN}Plugins installed. Rescan in your DAW if needed.${NC}"
echo "  VST3: $VST3_DIR"
echo "  CLAP: $CLAP_DIR"
echo ""
