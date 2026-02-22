#!/usr/bin/env bash
# =============================================================================
# Generate Xcode project from CMake
# =============================================================================
# Creates an Xcode .xcodeproj that uses Xcode's SDK and toolchain.
# This can help avoid SDK 26.2+ compatibility issues when building.
#
# Usage:
#   ./scripts/generate_xcode_project.sh [Release|Debug]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_TYPE="${1:-Release}"

XCODE_BUILD_DIR="$ROOT/build-xcode"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Generate Xcode Project from CMake                           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$ROOT"

# Check for Xcode (not just Command Line Tools)
if ! command -v xcodebuild &>/dev/null || ! xcodebuild -version &>/dev/null 2>&1; then
  echo -e "${YELLOW}Xcode not found. Only Command Line Tools detected.${NC}"
  echo ""
  echo "To generate an Xcode project, you need:"
  echo "  1. Install Xcode from the App Store (free, ~15GB)"
  echo "  2. Run: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
  echo "  3. Run this script again"
  echo ""
  echo "Alternatively, build with CMake + Make/Ninja:"
  echo "  cd KmiDi-compile"
  echo "  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_PLUGINS=ON"
  echo "  cmake --build build -j8"
  exit 1
fi

# Verify Xcode is active (not just CLT)
if xcode-select -p 2>/dev/null | grep -q "CommandLineTools"; then
  echo -e "${YELLOW}Command Line Tools active. Switching to Xcode...${NC}"
  if [ -d "/Applications/Xcode.app" ]; then
    sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    echo -e "${GREEN}Switched to Xcode${NC}"
  else
    echo -e "${YELLOW}Xcode.app not found in /Applications${NC}"
    echo "Install Xcode from the App Store, then run:"
    echo "  sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
    exit 1
  fi
fi

XCODE_VERSION=$(xcodebuild -version 2>&1 | head -1)
echo -e "${GREEN}Found: $XCODE_VERSION${NC}"

# Get Xcode SDK path
SDK_PATH=$(xcodebuild -version -sdk macosx Path 2>/dev/null || echo "")
if [ -z "$SDK_PATH" ]; then
  echo -e "${YELLOW}Could not detect Xcode SDK path. Using default.${NC}"
else
  echo -e "${GREEN}SDK: $SDK_PATH${NC}"
fi

# Qt6 path
QT_PREFIX=""
if command -v brew &>/dev/null; then
  QT_PREFIX="$(brew --prefix qt@6 2>/dev/null)" || true
fi
[ -n "$QT_PREFIX" ] && export CMAKE_PREFIX_PATH="$QT_PREFIX"

echo ""
echo -e "${YELLOW}Generating Xcode project ($BUILD_TYPE)...${NC}"
echo "  Build directory: $XCODE_BUILD_DIR"
echo ""

echo -e "${YELLOW}Applying local JUCE patches...${NC}"
"$ROOT/scripts/juce/apply_patches.sh"

# Generate Xcode project
rm -rf "$XCODE_BUILD_DIR"
mkdir -p "$XCODE_BUILD_DIR"

cmake -B "$XCODE_BUILD_DIR" -S . \
  -G Xcode \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DBUILD_PLUGINS=ON \
  -DBUILD_DESKTOP=ON \
  -DBUILD_KELLY_CORE=ON

XCODE_PROJ=$(find "$XCODE_BUILD_DIR" -maxdepth 1 -name "*.xcodeproj" -type d | head -1)

if [ -n "$XCODE_PROJ" ]; then
  echo ""
  echo -e "${GREEN}✓ Xcode project generated${NC}"
  echo ""
  echo "  Project: $XCODE_PROJ"
  echo ""
  echo "  To open in Xcode:"
  echo "    open \"$XCODE_PROJ\""
  echo ""
  echo "  To build from command line:"
  echo "    cd \"$XCODE_BUILD_DIR\""
  echo "    xcodebuild -project \"$(basename "$XCODE_PROJ")\" -scheme Kelly -configuration $BUILD_TYPE"
  echo ""
  echo "  Or build specific targets:"
  echo "    xcodebuild -project \"$(basename "$XCODE_PROJ")\" -target KellyPlugin_VST3 -configuration $BUILD_TYPE"
  echo "    xcodebuild -project \"$(basename "$XCODE_PROJ")\" -target KellyPlugin_CLAP -configuration $BUILD_TYPE"
  echo ""
else
  echo -e "${YELLOW}Xcode project not found in $XCODE_BUILD_DIR${NC}"
  exit 1
fi
