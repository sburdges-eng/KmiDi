#!/usr/bin/env bash
# =============================================================================
# Build Kelly - iDAW (Tauri) and create a macOS DMG installer
# =============================================================================
# Runs: npm run tauri build (with dmg bundle).
# Output: src-tauri/target/release/bundle/dmg/*.dmg
#
# Usage:
#   ./scripts/create_app_dmg.sh
#
# Optional: set COPY_DMG=1 to copy the DMG to project dist/ or workspace root.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Kelly - iDAW — Build & Create DMG                               ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$ROOT"

# Ensure Rust/cargo is on PATH (rustup installs to ~/.cargo/bin)
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

if ! command -v cargo &>/dev/null; then
  echo -e "${RED}Rust/cargo not found. Install Rust first:${NC}"
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  echo "  Then run this script again (or run: source \"\$HOME/.cargo/env\")."
  exit 1
fi

if [ ! -f "package.json" ] || [ ! -d "src-tauri" ]; then
  echo -e "${RED}Not in KmiDi-compile root (package.json / src-tauri missing).${NC}"
  exit 1
fi

echo -e "${YELLOW}Building Tauri app (frontend + Rust) and DMG...${NC}"
npm run tauri build -- --bundles dmg

BUNDLE_DIR="src-tauri/target/release/bundle"
DMG_DIR="$BUNDLE_DIR/dmg"
DMG_FILE=""

if [ -d "$DMG_DIR" ]; then
  DMG_FILE=$(find "$DMG_DIR" -maxdepth 1 -name "*.dmg" -type f 2>/dev/null | head -1)
fi
if [ -z "$DMG_FILE" ]; then
  DMG_FILE=$(find "$BUNDLE_DIR" -maxdepth 2 -name "*.dmg" -type f 2>/dev/null | head -1)
fi

if [ -n "$DMG_FILE" ] && [ -f "$DMG_FILE" ]; then
  echo ""
  echo -e "${GREEN}DMG created:${NC} $ROOT/$DMG_FILE"
  if [ "${COPY_DMG:-0}" = "1" ]; then
    mkdir -p "$ROOT/dist"
    cp "$DMG_FILE" "$ROOT/dist/"
    echo -e "${GREEN}Copied to:${NC} $ROOT/dist/$(basename "$DMG_FILE")"
  fi
else
  echo -e "${YELLOW}DMG not found under $BUNDLE_DIR. Build may have produced .app only.${NC}"
  APP_FILE=$(find "$BUNDLE_DIR" -maxdepth 2 -name "*.app" -type d 2>/dev/null | head -1)
  if [ -n "$APP_FILE" ]; then
    echo -e "App bundle: $ROOT/$APP_FILE"
    echo "Create DMG manually, e.g.:"
    echo "  mkdir -p /tmp/dmg_stage && cp -R \"$APP_FILE\" /tmp/dmg_stage/"
    echo "  ln -s /Applications /tmp/dmg_stage/Applications"
    echo "  hdiutil create -volname \"Kelly - iDAW\" -srcfolder /tmp/dmg_stage -ov -format UDZO \"Kelly-iDAW.dmg\""
  fi
  exit 1
fi
echo ""
