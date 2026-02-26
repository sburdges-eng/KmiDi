#!/usr/bin/env bash
# Build Kelly Listening Contract plugin
# Usage: ./scripts/build.sh [--with-core]
#   --with-core: link minimal_listening_core for constraint validation

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ROOT/build"

CORE_OPT=""
if [[ "${1:-}" == "--with-core" ]]; then
    CORE_OPT="-DKELLY_USE_CORE_BRIDGE=ON"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. $CORE_OPT
cmake --build . --config Release

echo ""
echo "Build complete. Plugin in: $BUILD_DIR/KellyListeningContract_artefacts/Release/"
